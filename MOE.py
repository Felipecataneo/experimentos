import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Configuração da página
st.set_page_config(
    page_title="🔬 Experimento: Transformer com Mixture-of-Experts (MoE)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# COMPONENTES DO MODELO (VERSÃO REVISADA E MELHORADA)
# =============================================================================

class SparseAttention(nn.Module):
    """
    Atenção com esparsificação aprendível, removendo instabilidades.
    """
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, 'A dimensão deve ser divisível pelo número de heads.'
        
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Limiar de esparsificação se torna um parâmetro aprendível.
        self.sparsity_threshold = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, self.dim_head).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Mecanismo de esparsificação estável:
        # Pontuações abaixo do limiar são mascaradas com um valor muito baixo.
        dots = torch.where(dots < self.sparsity_threshold, torch.tensor(-1e9, device=x.device), dots)

        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.to_out(out)

class ExpertModule(nn.Module):
    """Um único "expert": uma rede feed-forward padrão."""
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """
    Camada Mixture-of-Experts (MoE) com roteamento esparso.
    """
    def __init__(self, dim, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Lista de experts
        self.experts = nn.ModuleList([ExpertModule(dim, dropout=dropout) for _ in range(num_experts)])
        
        # Rede de gating para roteamento
        self.gating = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        x_flat = x.view(-1, d) # (b*n, d)

        # 1. Obter logits do gating para cada token
        gating_logits = self.gating(x_flat) # (b*n, num_experts)
        
        # 2. Calcular perda de balanceamento (auxiliary loss)
        # Incentiva o gating a usar todos os experts de forma equilibrada
        gating_probs = F.softmax(gating_logits, dim=-1)
        expert_usage = gating_probs.mean(0)
        expert_prob_dist = gating_probs.sum(0)
        
        load_balancing_loss = self.num_experts * torch.sum(expert_usage * expert_prob_dist)
        
        # 3. Roteamento: Encontrar os top_k experts para cada token
        top_k_weights, top_k_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1) # (b*n, top_k)

        # 4. Processamento esparso
        output = torch.zeros_like(x_flat)
        # Iterar pelos experts e processar apenas os tokens roteados para eles
        for i, expert in enumerate(self.experts):
            # Encontra quais tokens têm este expert em seu top_k
            token_indices, expert_rank = torch.where(top_k_indices == i)
            
            if token_indices.shape[0] > 0:
                # Pega os tokens e seus pesos de gating correspondentes
                tokens_for_expert = x_flat[token_indices]
                gating_weights_for_expert = top_k_weights[token_indices, expert_rank]
                
                # Processa os tokens pelo expert
                expert_output = expert(tokens_for_expert)
                
                # Pondera a saída e adiciona ao resultado final
                weighted_output = expert_output * gating_weights_for_expert.unsqueeze(-1)
                output.index_add_(0, token_indices, weighted_output)

        return output.view(b, n, d), load_balancing_loss

class MoETransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.attn = SparseAttention(dim, heads, dropout)
        self.ffn = MoELayer(dim, num_experts, top_k, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Atenção com conexão residual
        x = x + self.attn(self.norm1(x))
        
        # Mixture-of-Experts com conexão residual
        ffn_out, aux_loss = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x, aux_loss

class MoETransformerClassifier(nn.Module):
    """
    Classificador completo usando blocos de Transformer com MoE.
    """
    def __init__(self, vocab_size, dim, depth, heads, num_classes, num_experts, top_k, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        self.blocks = nn.ModuleList([
            MoETransformerBlock(dim, heads, num_experts, top_k) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        tokens = self.token_emb(x)
        b, n, d = tokens.shape
        
        x = tokens + self.pos_emb[:, :n]
        x = self.dropout(x)
        
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Pooling do token [CLS] (primeiro token da sequência)
        cls_token_output = x[:, 0]
        
        pooled = self.norm(cls_token_output)
        logits = self.to_logits(pooled)
        
        return logits, total_aux_loss / len(self.blocks) # Retorna logits e perda de balanceamento média

# =============================================================================
# DATASET E FUNÇÕES DE TREINAMENTO
# =============================================================================

@st.cache_data
def create_moe_dataset(max_features=5000, max_len=128):
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)
    X_test_tfidf = vectorizer.transform(newsgroups_test.data)

    # O ID 0 é para padding, o vocabulário real começa em 1.
    vocab_size = max_features + 1  
    cls_token_id = vocab_size # ID especial para o token [CLS]
    final_vocab_size = vocab_size + 1

    def to_sequence(X, max_len):
        sequences = []
        for i in range(X.shape[0]):
            row = X[i].toarray().flatten()
            top_indices = np.argsort(row)[-max_len:]
            top_indices = top_indices[row[top_indices] > 0]
            
            # Adiciona 1 aos índices para reservar 0 para padding
            seq = (top_indices + 1).tolist()
            
            # Truncar se necessário
            if len(seq) > max_len:
                seq = seq[:max_len]
            
            sequences.append(seq)
        return sequences

    X_train_seq = to_sequence(X_train_tfidf, max_len - 1) # -1 para deixar espaço para [CLS]
    X_test_seq = to_sequence(X_test_tfidf, max_len - 1)

    # Adicionar [CLS] e fazer padding
    def pad_and_add_cls(sequences, max_len, cls_id):
        padded_sequences = []
        for seq in sequences:
            # Adiciona [CLS] no início
            padded_seq = [cls_id] + seq
            # Faz padding com 0s
            padding_len = max_len - len(padded_seq)
            padded_seq.extend([0] * padding_len)
            padded_sequences.append(padded_seq)
        return torch.tensor(padded_sequences, dtype=torch.long)

    X_train_tensor = pad_and_add_cls(X_train_seq, max_len, cls_token_id)
    X_test_tensor = pad_and_add_cls(X_test_seq, max_len, cls_token_id)

    y_train = torch.tensor(newsgroups_train.target, dtype=torch.long)
    y_test = torch.tensor(newsgroups_test.target, dtype=torch.long)

    return (X_train_tensor, y_train), (X_test_tensor, y_test), final_vocab_size

def run_training_experiment(config, X_train, y_train, X_test, y_test, vocab_size, progress_callback=None):
    model = MoETransformerClassifier(
        vocab_size=vocab_size,
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        num_classes=4,
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        max_seq_len=config['max_len']
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    metrics = {'train_losses': [], 'train_accs': [], 'aux_losses': []}
    
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss, epoch_acc, epoch_aux_loss, batches = 0, 0, 0, 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            logits, aux_loss = model(batch_x)
            main_loss = F.cross_entropy(logits, batch_y)
            total_loss = main_loss + config['load_balancer_alpha'] * aux_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += main_loss.item()
            epoch_aux_loss += aux_loss.item()
            acc = (logits.argmax(dim=-1) == batch_y).float().mean()
            epoch_acc += acc.item()
            batches += 1
        
        metrics['train_losses'].append(epoch_loss / batches)
        metrics['train_accs'].append(epoch_acc / batches)
        metrics['aux_losses'].append(epoch_aux_loss / batches)
        
        if progress_callback:
            progress_callback(epoch + 1, config['epochs'])
    
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(X_test)
        test_acc = (test_logits.argmax(dim=-1) == y_test).float().mean()
        
    metrics['test_acc'] = test_acc.item()
    return metrics, model

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    st.title("🔬 Experimento: Transformer com Mixture-of-Experts (MoE)")
    st.markdown("*Uma arquitetura moderna e esparsa para classificação de texto.*")
    
    st.sidebar.header("⚙️ Configurações do Experimento")
    st.sidebar.subheader("🧠 Arquitetura do Modelo")
    dim = st.sidebar.slider("Dimensão do modelo (dim)", 64, 256, 128, 32)
    depth = st.sidebar.slider("Profundidade (camadas)", 1, 6, 2)
    heads = st.sidebar.slider("Heads de Atenção", 2, 8, 4, 2)
    
    st.sidebar.subheader("🧩 Configurações MoE")
    num_experts = st.sidebar.slider("Número de Experts", 4, 16, 8, 2)
    top_k = st.sidebar.slider("Top-K Experts a usar", 1, 4, 2)

    st.sidebar.subheader("🎯 Parâmetros de Treinamento")
    epochs = st.sidebar.slider("Épocas", 5, 50, 20)
    batch_size = st.sidebar.slider("Batch size", 16, 128, 64, 16)
    lr = st.sidebar.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    load_balancer_alpha = st.sidebar.slider("Peso da Perda de Balanceamento (α)", 0.0, 0.1, 0.01, 0.005, format="%.3f")

    if st.sidebar.button("🚀 Iniciar Experimento MoE"):
        config = {
            'dim': dim, 'depth': depth, 'heads': heads,
            'num_experts': num_experts, 'top_k': top_k,
            'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
            'load_balancer_alpha': load_balancer_alpha,
            'max_len': 128
        }
        run_experiment(config)

def run_experiment(config):
    st.header("🔬 Executando Experimento...")
    
    with st.spinner("📊 Carregando e pré-processando dataset..."):
        (X_train, y_train), (X_test, y_test), vocab_size = create_moe_dataset(max_len=config['max_len'])
        st.success(f"✅ Dataset carregado: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} de teste. Vocabulário: {vocab_size}")
    
    st.subheader(f"🔬 Treinando Modelo MoE...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(current_epoch, total_epochs):
        progress = current_epoch / total_epochs
        progress_bar.progress(progress)
        status_text.text(f"Época {current_epoch}/{total_epochs}")
    
    with st.spinner(f"Treinando..."):
        metrics, model = run_training_experiment(
            config, X_train, y_train, X_test, y_test, vocab_size, progress_callback
        )
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Treinamento concluído! Acurácia final no teste: {metrics['test_acc']:.4f}")
    
    visualize_results(metrics, config)

def visualize_results(metrics, config):
    st.header("📊 Resultados do Experimento")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Acurácia Final (Teste)", f"{metrics['test_acc']:.4f}")
    col2.metric("📉 Loss Final (Treino)", f"{metrics['train_losses'][-1]:.4f}")
    col3.metric("⚖️ Loss de Balanceamento", f"{metrics['aux_losses'][-1]:.4f}")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("📉 Perda de Classificação (Treino)", "📈 Acurácia de Treinamento", 
                       "⚖️ Perda de Balanceamento (MoE)", "🎯 Acurácia Final (Teste)"),
        specs=[[{}, {}], [{}, {"type": "indicator"}]]
    )
    
    epochs_range = list(range(1, config['epochs'] + 1))
    
    # Loss de Treinamento
    fig.add_trace(go.Scatter(x=epochs_range, y=metrics['train_losses'], name="Loss (Treino)"), row=1, col=1)
    # Acurácia de Treinamento
    fig.add_trace(go.Scatter(x=epochs_range, y=metrics['train_accs'], name="Acurácia (Treino)"), row=1, col=2)
    # Loss de Balanceamento
    fig.add_trace(go.Scatter(x=epochs_range, y=metrics['aux_losses'], name="Loss de Balanceamento"), row=2, col=1)

    # Indicador de Acurácia Final
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['test_acc'] * 100,
        title={'text': "Acurácia de Teste (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 100]}}
    ), row=2, col=2)

    fig.update_layout(height=700, title_text="🔬 Análise do Treinamento do Modelo MoE", showlegend=False)
    fig.update_xaxes(title_text="Época", row=1, col=1)
    fig.update_xaxes(title_text="Época", row=1, col=2)
    fig.update_xaxes(title_text="Época", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Análise e Próximos Passos")
    st.markdown(f"""
    O modelo treinado com **{config['num_experts']} experts** e roteamento **top-{config['top_k']}** alcançou uma performance promissora.

    - A **Perda de Classificação** deve diminuir consistentemente, indicando que o modelo está aprendendo a tarefa principal.
    - A **Acurácia de Treinamento** deve aumentar, mostrando a capacidade do modelo de se ajustar aos dados.
    - A **Perda de Balanceamento** idealmente se mantém baixa e estável. Um valor muito alto ou crescente pode indicar que a rede de gating não está distribuindo a carga de trabalho entre os experts de forma eficaz. O peso `α = {config['load_balancer_alpha']}` é crucial para controlar isso.
    
    **Sugestões para exploração:**
    1.  **Ajustar o Hiperparâmetro `α`:** Um `α` maior força um balanceamento mais rigoroso, mas pode prejudicar o aprendizado da tarefa principal. Encontrar o ponto ideal é chave.
    2.  **Variar o Número de Experts e `k`:** Mais experts aumentam a capacidade do modelo, mas podem exigir mais dados ou um `α` mais alto para treinar bem. Usar `k=1` é mais rápido, enquanto `k=2` permite uma combinação mais rica de conhecimento.
    3.  **Analisar a Especialização dos Experts:** Uma análise avançada seria investigar quais tipos de tokens ou sentenças são roteados para cada expert, para entender se eles estão se especializando em tópicos (ex: um expert para 'religião', outro para 'tecnologia').
    """)

if __name__ == "__main__":
    main()