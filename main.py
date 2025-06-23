import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import math

# Configuração da página
st.set_page_config(
    page_title="🔬 Quantum-Inspired Deep Learning v3.1",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# COMPONENTES QUANTUM-INSPIRED v3.1
# =============================================================================

class QuantumPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EnhancedUncertaintyAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, uncertainty_samples=4):
        super().__init__()
        self.heads, self.dim_head = heads, dim // heads
        self.scale, self.uncertainty_samples = self.dim_head**-0.5, uncertainty_samples
        self.q_projections = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(uncertainty_samples)])
        self.k_projections = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(uncertainty_samples)])
        self.v_projections = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(uncertainty_samples)])
        self.coherence_gate = nn.Sequential(nn.Linear(dim, uncertainty_samples), nn.Sigmoid())
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.hypothesis_weights = nn.Parameter(torch.ones(uncertainty_samples))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, collapse_uncertainty=False):
        b, n, h = x.shape[0], x.shape[1], self.heads
        if self.training and not collapse_uncertainty:
            outputs, attention_maps = [], []
            coherence_weights = self.coherence_gate(x.mean(dim=1))
            for i in range(self.uncertainty_samples):
                q = self.q_projections[i](x).view(b, n, h, self.dim_head).transpose(1, 2)
                k = self.k_projections[i](x).view(b, n, h, self.dim_head).transpose(1, 2)
                v = self.v_projections[i](x).view(b, n, h, self.dim_head).transpose(1, 2)
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                attn = F.softmax(dots / torch.clamp(self.temperature, 0.1, 10), dim=-1)
                out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, -1)
                out = out * coherence_weights[:, i].view(b, 1, 1)
                outputs.append(out)
                attention_maps.append(attn)
            global_weights = F.softmax(self.hypothesis_weights / torch.clamp(self.temperature, 0.1, 10), dim=0)
            combined_output = sum(w * out for w, out in zip(global_weights, outputs))
            combined_attention = sum(w * attn for w, attn in zip(global_weights, attention_maps))
            return self.to_out(combined_output), combined_attention.mean(dim=1)
        else:
            best_idx = torch.argmax(self.hypothesis_weights).item()
            q = self.q_projections[best_idx](x).view(b, n, h, self.dim_head).transpose(1, 2)
            k = self.k_projections[best_idx](x).view(b, n, h, self.dim_head).transpose(1, 2)
            v = self.v_projections[best_idx](x).view(b, n, h, self.dim_head).transpose(1, 2)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, -1)
            return self.to_out(out), attn.mean(dim=1)

class QuantumInterferenceFFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1, num_paths=3):
        super().__init__()
        inner_dim, self.num_paths = int(dim * mult), num_paths
        self.paths = nn.ModuleList()
        for i in range(num_paths):
            activation = nn.GELU() if i % 3 == 0 else nn.SiLU() if i % 3 == 1 else nn.ReLU()
            self.paths.append(nn.Sequential(nn.Linear(dim, inner_dim), activation, nn.Dropout(dropout), nn.Linear(inner_dim, dim), nn.Dropout(dropout)))
        self.interference_weights = nn.Parameter(torch.randn(num_paths))
        self.phase_shifts = nn.Parameter(torch.zeros(num_paths))

    def forward(self, x, training_step=0):
        outputs = []
        for i, path in enumerate(self.paths):
            path_output = path(x)
            phase = self.phase_shifts[i] + training_step * 0.001 * (i + 1)
            outputs.append(path_output * torch.cos(phase))
        weights = F.softmax(self.interference_weights, dim=0)
        return sum(w * out for w, out in zip(weights, outputs))

class QuantumResidualBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn, self.ffn = EnhancedUncertaintyAttention(dim, heads, dropout), QuantumInterferenceFFN(dim, dropout=dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.collapse_scheduler = nn.Parameter(torch.tensor(-2.0))
        self.attn_gate, self.ffn_gate = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
    def forward(self, x, training_step=0, force_collapse=False):
        collapse_prob = torch.sigmoid(self.collapse_scheduler + (training_step / 5000))
        collapse = force_collapse or (self.training and torch.rand(1).item() < collapse_prob)
        attn_out, attn_weights = self.attn(self.norm1(x), collapse_uncertainty=collapse)
        x = x + torch.sigmoid(self.attn_gate) * attn_out
        ffn_out = self.ffn(self.norm2(x), training_step)
        x = x + torch.sigmoid(self.ffn_gate) * ffn_out
        return x, attn_weights

class QuantumClassifierV3(nn.Module):
    def __init__(self, vocab_size, dim=256, depth=6, num_classes=4, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0) # Informa que 0 é padding
        self.pos_encoding = QuantumPositionalEncoding(dim, max_seq_len)
        self.blocks = nn.ModuleList([QuantumResidualBlock(dim, heads=max(4, dim // 32), dropout=dropout) for _ in range(depth)])
        self.global_pool, self.max_pool = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
        self.norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout * 1.5), nn.Linear(dim, num_classes))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding) and m.padding_idx is not None:
                with torch.no_grad(): m.weight[m.padding_idx].fill_(0)
    def forward(self, x, training_step=0):
        x = self.pos_encoding(self.token_emb(x))
        attention_maps = []
        for i, block in enumerate(self.blocks):
            force_collapse = (not self.training) and (i >= len(self.blocks) - 1)
            x, attn = block(x, training_step, force_collapse)
            if attn is not None: attention_maps.append(attn)
        x_transposed = x.transpose(1, 2)
        avg_pooled, max_pooled = self.global_pool(x_transposed).squeeze(-1), self.max_pool(x_transposed).squeeze(-1)
        pooled = self.norm(torch.cat([avg_pooled, max_pooled], dim=-1))
        return self.classifier(pooled), attention_maps

class QuantumOptimizerV3:
    def __init__(self, model, lr=0.001, weight_decay=0.01):
        self.model = model
        self.base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-8)
        self.step_count, self.warmup_steps, self.initial_lr = 0, 500, lr
    def step(self, loss):
        self.step_count += 1
        self.base_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if self.step_count % 20 == 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None and 'weight' in name:
                        wave_modulation = 1 + 0.02 * np.sin(self.step_count * 0.05 + (hash(name) % 1000) * 0.00628)
                        param.grad.data.mul_(wave_modulation)
        if self.step_count <= self.warmup_steps:
            for pg in self.base_optimizer.param_groups: pg['lr'] = self.initial_lr * (self.step_count / self.warmup_steps)
        self.base_optimizer.step()
    @property
    def param_groups(self): return self.base_optimizer.param_groups

@st.cache_data
def create_enhanced_dataset_v3(max_features=12000, max_len=256):
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc']
    train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), max_df=0.7, min_df=3, sublinear_tf=True)
    X_train_tfidf, X_test_tfidf = vectorizer.fit_transform(train.data), vectorizer.transform(test.data)
    vocab_size = len(vectorizer.get_feature_names_out())
    
    # ================== LÓGICA DE PADDING CORRIGIDA ==================
    def to_sequence(X, max_len):
        sequences = []
        for i in range(X.shape[0]):
            row = X[i].toarray().flatten()
            # Pega os índices dos tokens com score > 0 e os ordena
            all_sorted_indices = np.argsort(row)[::-1]
            valid_indices = all_sorted_indices[row[all_sorted_indices] > 0]
            
            # Trunca se necessário
            truncated_indices = valid_indices[:max_len]
            
            # Faz o padding com o ID 0
            padded = np.pad(
                truncated_indices,
                (0, max_len - len(truncated_indices)),
                'constant',
                constant_values=-1 # Placeholder para ser 0 depois do +1
            )
            
            # Soma 1 a todos os IDs. O padding (-1) se torna 0. Os tokens (0,1,2...) se tornam (1,2,3...).
            sequences.append(padded + 1)
            
        return torch.tensor(sequences, dtype=torch.long)
    # =================================================================

    return (to_sequence(X_train_tfidf, max_len), torch.tensor(train.target, dtype=torch.long)), \
           (to_sequence(X_test_tfidf, max_len), torch.tensor(test.target, dtype=torch.long)), \
           vocab_size + 1, len(categories)

def run_quantum_experiment_v3(config, X_train, y_train, X_test, y_test, vocab_size, num_classes, progress_callback=None):
    model = QuantumClassifierV3(vocab_size=vocab_size, dim=config['dim'], depth=config['depth'], num_classes=num_classes, max_seq_len=256, dropout=config['dropout'])
    if config['optimizer_type'] == 'quantum': optimizer = QuantumOptimizerV3(model, lr=config['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], epochs=config['epochs'], steps_per_epoch=len(X_train) // config['batch_size'] + 1, pct_start=0.1)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    metrics = {'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': [], 'attention_entropies': [], 'learning_rates': []}
    best_val_acc, patience, patience_counter = 0, 7, 0
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss, epoch_acc, epoch_batches, epoch_entropy = 0, 0, 0, 0
        indices = torch.randperm(len(X_train_split))
        for i in range(0, len(X_train_split), config['batch_size']):
            batch_indices = indices[i:i+config['batch_size']]
            batch_x, batch_y = X_train_split[batch_indices], y_train_split[batch_indices]
            training_step = epoch * (len(X_train_split) // config['batch_size']) + (i // config['batch_size'])
            logits, attention_maps = model(batch_x, training_step)
            loss = F.cross_entropy(logits, batch_y, label_smoothing=0.1)
            if config['optimizer_type'] == 'quantum': optimizer.step(loss)
            else: optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()
            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(dim=-1) == batch_y).float().mean().item()
            if attention_maps: epoch_entropy += sum(-(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item() for attn in attention_maps) / len(attention_maps)
            epoch_batches += 1
        model.eval()
        val_loss, val_acc, val_batches = 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(X_val), config['batch_size']):
                batch_x, batch_y = X_val[i:i+config['batch_size']], y_val[i:i+config['batch_size']]
                logits, _ = model(batch_x); val_loss += F.cross_entropy(logits, batch_y).item(); val_acc += (logits.argmax(dim=-1) == batch_y).float().mean().item(); val_batches += 1
        val_acc_avg = val_acc / val_batches if val_batches > 0 else 0
        metrics['train_losses'].append(epoch_loss/epoch_batches); metrics['train_accs'].append(epoch_acc/epoch_batches)
        metrics['val_losses'].append(val_loss/val_batches); metrics['val_accs'].append(val_acc_avg)
        metrics['attention_entropies'].append(epoch_entropy / epoch_batches if epoch_batches > 0 else 0)
        metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        if val_acc_avg > best_val_acc: best_val_acc, patience_counter = val_acc_avg, 0
        else: patience_counter += 1
        if patience_counter >= patience: st.info(f"Early stopping na época {epoch+1}"); break
        if progress_callback: progress_callback(epoch + 1, config['epochs'])
    model.eval()
    with torch.no_grad(): test_acc = (model(X_test)[0].argmax(dim=-1) == y_test).float().mean().item()
    metrics['test_acc'] = test_acc
    return metrics, model

def main():
    st.title("🔬 Quantum-Inspired Deep Learning v3.1")
    st.markdown("*Otimizado para alcançar >80% de acurácia mantendo inovação quântica*")
    with st.expander("🚀 Melhorias v3.1"):
        st.markdown("""**Principais melhorias:**\n- **Correção Crítica:** Lógica de padding de dados robusta para evitar `ValueError`.\n- **Arquitetura:** Codificação posicional, atenção com gate de coerência, FFN com múltiplos caminhos, pooling multi-escala.\n- **Dados:** TF-IDF com n-grams, sequências mais longas, mais features.\n- **Treinamento:** Label smoothing, OneCycleLR scheduler, early stopping, warmup adaptativo.""")
    st.sidebar.header("⚙️ Configurações v3.1")
    st.sidebar.subheader("🧠 Arquitetura")
    dim = st.sidebar.slider("Dimensão", 128, 512, 256, 64)
    depth = st.sidebar.slider("Profundidade", 3, 8, 4)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    st.sidebar.subheader("🎯 Treinamento")
    epochs = st.sidebar.slider("Épocas", 10, 100, 40)
    batch_size = st.sidebar.slider("Batch size", 16, 64, 32, 16)
    lr = st.sidebar.slider("Max Learning rate", 0.0001, 0.01, 0.0015, 0.0001, format="%.4f")
    experiment_type = st.sidebar.selectbox("Experimento:", ["Comparativo v3.1", "Quantum v3.1 Only", "Classical v3.1 Only"])
    if st.sidebar.button("🚀 Executar Experimento v3.1"):
        run_enhanced_experiment_ui_v3(dim, depth, dropout, epochs, batch_size, lr, experiment_type)

def run_enhanced_experiment_ui_v3(dim, depth, dropout, epochs, batch_size, lr, experiment_type):
    st.header("🔬 Executando Experimento v3.1...")
    with st.spinner("📊 Preparando dataset v3.1..."):
        (X_train, y_train), (X_test, y_test), vocab_size, num_classes = create_enhanced_dataset_v3()
        st.success(f"✅ Dataset: {X_train.shape[0]} treino, {X_test.shape[0]} teste, {num_classes} classes, Vocab: {vocab_size}")
    configs = {}
    base_config = {'dim': dim, 'depth': depth, 'dropout': dropout, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr}
    if experiment_type in ["Comparativo v3.1", "Quantum v3.1 Only"]:
        configs['quantum_v3'] = {**base_config, 'name': 'Quantum-Inspired v3.1', 'optimizer_type': 'quantum', 'color': '#636EFA'}
    if experiment_type in ["Comparativo v3.1", "Classical v3.1 Only"]:
        configs['classical_v3'] = {**base_config, 'name': 'Classical Transformer v3.1', 'optimizer_type': 'classical', 'color': '#FFA15A'}
    results = {}
    for name, config in configs.items():
        st.subheader(f"🔬 Treinando {config['name']}...")
        progress_bar, status_text = st.progress(0), st.empty()
        def cb(curr, total): progress_bar.progress(curr/total); status_text.text(f"Época {curr}/{total}")
        with st.spinner("..."):
            metrics, _ = run_quantum_experiment_v3(config, X_train, y_train, X_test, y_test, vocab_size, num_classes, cb)
        results[name] = {**metrics, 'config': config}
        progress_bar.empty(); status_text.empty()
        st.success(f"✅ {config['name']} concluído! Acurácia no teste: {metrics['test_acc']:.4f}")
    visualize_enhanced_results_v3(results)

def visualize_enhanced_results_v3(results):
    st.header("📊 Resultados v3.1")
    cols = st.columns(len(results)); i = 0
    for name, data in results.items():
        with cols[i]: st.metric(f"🎯 {data['config']['name']} (Teste)", f"{data['test_acc']:.4f}", f"Melhor Val: {max(data['val_accs']):.4f}"); i+=1
    fig = make_subplots(rows=2, cols=2, subplot_titles=("📉 Loss (Treino vs Val)", "📈 Acurácia (Treino vs Val)", "📚 Learning Rate", "🎯 Comparação Final"), vertical_spacing=0.2)
    epochs = lambda data: list(range(1, len(data['train_losses']) + 1))
    for name, data in results.items():
        config = data['config']
        fig.add_trace(go.Scatter(x=epochs(data), y=data['train_losses'], name=f"{config['name']} Train", line=dict(color=config['color']), legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs(data), y=data['val_losses'], name=f"{config['name']} Val", line=dict(color=config['color'], dash='dash'), legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs(data), y=data['train_accs'], name=f"{config['name']} Train", line=dict(color=config['color']), legendgroup=name, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs(data), y=data['val_accs'], name=f"{config['name']} Val", line=dict(color=config['color'], dash='dash'), legendgroup=name, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs(data), y=data['learning_rates'], name=f"LR {config['name']}", line=dict(color=config['color'])), row=2, col=1)
    final_accs = [data['test_acc'] for data in results.values()]
    model_names = [data['config']['name'] for data in results.values()]
    colors = [data['config']['color'] for data in results.values()]
    fig.add_trace(go.Bar(x=model_names, y=final_accs, marker_color=colors, text=[f"{acc:.4f}" for acc in final_accs], textposition='auto'), row=2, col=2)
    fig.update_layout(height=800, title_text="🔬 Análise Detalhada do Experimento v3.1", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()