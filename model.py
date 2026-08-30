"""
Vision-Language Model from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_image_into_patches
import torch

def split_image_into_patches(image, patch_size):
    """Split an image tensor (B, C, H, W) into a sequence of (patch_size, patch_size) patches.

    Returns a tensor of shape (B, num_patches, C, patch_size, patch_size) in row-major order.
    """
    B, C, H, W = image.shape

    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    patches = patches.permute(0, 2, 3, 1, 4, 5)

    return patches.reshape(
        B,
        (H // patch_size) * (W // patch_size),
        C,
        patch_size,
        patch_size
    )

# Step 2 - flatten_patches
def flatten_patches(patches):
    # Flatten each patch's channel and spatial dimensions into one vector,
    # while keeping the (B, num_patches) leading dimensions.
    return patches.reshape(patches.shape[0], patches.shape[1], -1)

# Step 3 - linear_projection
def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    return x @ weight.T + bias

# Step 4 - project_patches_to_embeddings
def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    # Linearly project flattened image patches into the ViT embedding dimension.
    return linear_projection(flat_patches, patch_proj_weight, patch_proj_bias)

# Step 5 - prepend_class_token
def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    batch_size = patch_embeddings.shape[0]
    cls_tokens = class_token.expand(batch_size, -1, -1)
    return torch.cat([cls_tokens, patch_embeddings], dim=1)

# Step 6 - add_position_embeddings
def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    return tokens + position_embeddings

# Step 7 - compute_attention_scores
def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    return torch.matmul(q, k.transpose(-2, -1))

# Step 8 - scale_attention_scores
import math

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    return scores / math.sqrt(d_head)

# Step 9 - apply_attention_mask
def apply_attention_mask(scores, mask):
    # Add an additive mask (0 = allowed, -inf = blocked) to attention scores.
    if mask is None:
        return scores
    return scores + mask

# Step 10 - attention_softmax
def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    return torch.softmax(masked_scores, dim=-1)

# Step 11 - attention_context
def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    return torch.matmul(attn_weights, v)

# Step 12 - scaled_dot_product_attention
def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    scores = compute_attention_scores(q, k)
    scores = scale_attention_scores(scores, q.shape[-1])
    scores = apply_attention_mask(scores, mask)
    attn_weights = attention_softmax(scores)
    return attention_context(attn_weights, v)

# Step 13 - split_into_heads
def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    B, S, d_model = x.shape
    d_head = d_model // num_heads

    x = x.reshape(B, S, num_heads, d_head)
    return x.transpose(1, 2)

# Step 14 - merge_heads
def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    B, num_heads, S, d_head = x.shape

    x = x.transpose(1, 2).contiguous()
    return x.reshape(B, S, num_heads * d_head)

# Step 15 - project_qkv
def project_qkv(x, wq, bq, wk, bk, wv, bv):
    # Project x into separate query, key, and value tensors.
    q = linear_projection(x, wq, bq)
    k = linear_projection(x, wk, bk)
    v = linear_projection(x, wv, bv)

    return q, k, v

# Step 16 - split_qkv_into_heads
def split_qkv_into_heads(q, k, v, num_heads):
    # Reshape q, k, v into multi-head form.
    q_h = split_into_heads(q, num_heads)
    k_h = split_into_heads(k, num_heads)
    v_h = split_into_heads(v, num_heads)

    return q_h, k_h, v_h

# Step 17 - multi_head_attention_scores
def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 18 - merge_and_output_project
def merge_and_output_project(context_heads, wo, bo):
    """Merge heads back to d_model and apply the output projection."""
    merged = merge_heads(context_heads)
    return linear_projection(merged, wo, bo)

# Step 19 - multi_head_self_attention
def multi_head_self_attention(x, params, num_heads, mask=None):
    """Run full multi-head self-attention: QKV proj, head split, attention, merge, output proj."""
    
    q, k, v = project_qkv(
        x,
        params["wq"], params["bq"],
        params["wk"], params["bk"],
        params["wv"], params["bv"]
    )

    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)

    context_heads = multi_head_attention_scores(q_h, k_h, v_h, mask)

    return merge_and_output_project(
        context_heads,
        params["wo"],
        params["bo"]
    )

# Step 20 - gelu_activation
def gelu_activation(x):
    """Apply the exact (erf-based) GELU activation elementwise to x."""
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))

# Step 21 - mlp_first_layer
def mlp_first_layer(x, w1, b1):
    """Apply the first linear layer of the MLP block followed by GELU."""
    return gelu_activation(linear_projection(x, w1, b1))

# Step 22 - mlp_second_layer
def mlp_second_layer(h, w2, b2):
    # Project the MLP hidden activations back down to d_model.
    return linear_projection(h, w2, b2)

# Step 23 - mlp_block
def mlp_block(x, params):
    """Two-layer position-wise MLP with GELU between the layers."""
    h = mlp_first_layer(x, params["w1"], params["b1"])
    return mlp_second_layer(h, params["w2"], params["b2"])

# Step 24 - compute_layernorm_stats
def compute_layernorm_stats(x, eps=1e-5):
    # Return mean and biased (population) variance along the last dimension.
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return mean, var

# Step 25 - layer_norm
def layer_norm(x, gamma, beta, eps=1e-5):
    # Normalize the last dimension and apply learnable scale and shift.
    mean, var = compute_layernorm_stats(x, eps)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta

# Step 26 - residual_add
def residual_add(residual, sublayer_output):
    """Add residual skip connection to a sublayer's output."""
    return residual + sublayer_output

# Step 27 - pre_norm_sublayer
def pre_norm_sublayer(x, gamma, beta, sublayer_fn):
    """Apply pre-norm: LN(x) -> sublayer -> add residual x."""
    normalized = layer_norm(x, gamma, beta)
    sublayer_output = sublayer_fn(normalized)
    return residual_add(x, sublayer_output)

# Step 28 - vision_encoder_block
def vision_encoder_block(x, block_params, num_heads):
    # Pre-norm multi-head self-attention with residual.
    x = pre_norm_sublayer(
        x,
        block_params["ln1_gamma"],
        block_params["ln1_beta"],
        lambda t: multi_head_self_attention(
            t,
            block_params["attn"],
            num_heads
        )
    )

    # Pre-norm MLP with residual.
    x = pre_norm_sublayer(
        x,
        block_params["ln2_gamma"],
        block_params["ln2_beta"],
        lambda t: mlp_block(t, block_params["mlp"])
    )

    return x

# Step 29 - vision_encoder
def vision_encoder(patch_sequence, encoder_params, num_heads):
    """Stack ViT encoder blocks then apply a final layer norm to the patch sequence."""
    x = patch_sequence

    for block_params in encoder_params["blocks"]:
        x = vision_encoder_block(x, block_params, num_heads)

    return layer_norm(
        x,
        encoder_params["final_ln_gamma"],
        encoder_params["final_ln_beta"]
    )

# Step 30 - extract_patch_features
def extract_patch_features(encoder_output):
    """Drop the [CLS] token from a ViT encoder output."""
    return encoder_output[:, 1:, :]

# Step 31 - projector_first_layer
def projector_first_layer(patch_features, w1, b1):
    # Apply the first projector linear layer followed by GELU.
    hidden = patch_features @ w1 + b1
    return gelu_activation(hidden)

# Step 32 - projector_second_layer
def projector_second_layer(hidden, w2, b2):
    """Map hidden activations into the language model embedding space."""
    return hidden @ w2 + b2

# Step 33 - vision_language_projector
def vision_language_projector(patch_features, params):
    """Map patch features into the language model embedding space."""
    hidden = projector_first_layer(
        patch_features,
        params["w1"],
        params["b1"]
    )

    return projector_second_layer(
        hidden,
        params["w2"],
        params["b2"]
    )

# Step 34 - build_token_vocabulary
def build_token_vocabulary(texts, image_token="<image>", pad_token="<pad>"):
    # Collect all whitespace-separated tokens.
    tokens = set()

    for text in texts:
        tokens.update(text.split())

    # Ensure special tokens have fixed IDs and remaining tokens are sorted.
    tokens.discard(pad_token)
    tokens.discard(image_token)

    vocab = {
        pad_token: 0,
        image_token: 1,
    }

    for token in sorted(tokens):
        vocab[token] = len(vocab)

    return vocab

# Step 35 - encode_text_to_ids
def encode_text_to_ids(text, vocab):
    # Split text on whitespace and map each token to its vocabulary ID.
    return [vocab[token] for token in text.split()]

# Step 36 - embed_token_ids
def embed_token_ids(token_ids, embedding_matrix):
    """Look up embedding vectors for each token id."""
    return embedding_matrix[token_ids]

# Step 37 - add_text_position_embeddings
def add_text_position_embeddings(text_embeddings, position_embeddings):
    """Add learnable position embeddings to text token embeddings."""
    T = text_embeddings.shape[0]
    return text_embeddings + position_embeddings[:T]

# Step 38 - find_image_placeholder_positions
def find_image_placeholder_positions(token_ids, image_token_id):
    """Return a list of indices where token_ids == image_token_id."""
    return (token_ids == image_token_id).nonzero(as_tuple=False).flatten().tolist()

# Step 39 - insert_image_tokens
def insert_image_tokens(text_embeddings, image_tokens, placeholder_position):
    """Splice image tokens into the text embedding sequence at the placeholder slot."""
    before = text_embeddings[:placeholder_position]
    after = text_embeddings[placeholder_position + 1:]

    return torch.cat([before, image_tokens, after], dim=0)

# Step 40 - build_multimodal_embeddings
def build_multimodal_embeddings(
    token_ids,
    image_tokens,
    embedding_matrix,
    position_embeddings,
    image_token_id
):
    # Embed text tokens.
    text_embeddings = embed_token_ids(token_ids, embedding_matrix)

    # Add positional embeddings to the original text sequence.
    text_embeddings = add_text_position_embeddings(
        text_embeddings,
        position_embeddings
    )

    # Find the image placeholder and replace it with projected image tokens.
    positions = find_image_placeholder_positions(token_ids, image_token_id)
    placeholder_position = positions[0]

    return insert_image_tokens(
        text_embeddings,
        image_tokens,
        placeholder_position
    )

# Step 41 - build_label_tensor
def build_label_tensor(
    token_ids,
    image_token_id,
    pad_token_id,
    num_image_tokens,
    ignore_index=-100
):
    """Build the label tensor aligned to the fused multimodal sequence."""
    # Use the helper to locate image placeholder positions.
    image_positions = set(
        find_image_placeholder_positions(token_ids, image_token_id)
    )

    labels = []

    for i, token_id in enumerate(token_ids.tolist()):
        if i in image_positions:
            # Replace one image placeholder with num_image_tokens ignored positions.
            labels.extend([ignore_index] * num_image_tokens)
        elif token_id == pad_token_id:
            labels.append(ignore_index)
        else:
            labels.append(token_id)

    return torch.tensor(
        labels,
        dtype=torch.long,
        device=token_ids.device
    )

# Step 42 - build_causal_mask
def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) additive causal mask."""
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf")
    )
    return mask

# Step 43 - decoder_block
def decoder_block(x, params, causal_mask):
    # Add a batch dimension because multi_head_self_attention expects (B, S, d_model).
    x_batched = x.unsqueeze(0)

    # Pre-norm masked self-attention with residual.
    x_batched = pre_norm_sublayer(
        x_batched,
        params["ln1"]["gamma"],
        params["ln1"]["beta"],
        lambda t: multi_head_self_attention(
            t,
            params["attn"],
            params["num_heads"],
            mask=causal_mask
        )
    )

    # Pre-norm MLP with residual.
    x_batched = pre_norm_sublayer(
        x_batched,
        params["ln2"]["gamma"],
        params["ln2"]["beta"],
        lambda t: mlp_block(t, params["mlp"])
    )

    return x_batched.squeeze(0)

# Step 44 - language_model_decoder
def language_model_decoder(x, blocks_params, causal_mask):
    # Apply each decoder block sequentially.
    for params in blocks_params:
        x = decoder_block(x, params, causal_mask)

    return x

# Step 45 - final_layer_norm (not yet solved)
# TODO: implement

# Step 46 - language_model_head (not yet solved)
# TODO: implement

# Step 47 - encode_image_to_tokens (not yet solved)
# TODO: implement

# Step 48 - vision_language_forward (not yet solved)
# TODO: implement

# Step 49 - shift_logits_and_labels (not yet solved)
# TODO: implement

# Step 50 - per_position_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - masked_mean_loss (not yet solved)
# TODO: implement

# Step 52 - greedy_next_token (not yet solved)
# TODO: implement

# Step 53 - apply_temperature (not yet solved)
# TODO: implement

# Step 54 - top_k_filter (not yet solved)
# TODO: implement

# Step 55 - sample_from_logits (not yet solved)
# TODO: implement

# Step 56 - generate_caption (not yet solved)
# TODO: implement

# Step 57 - initialize_vlm_parameters (not yet solved)
# TODO: implement

# Step 58 - collect_parameters (not yet solved)
# TODO: implement

# Step 59 - zero_gradients (not yet solved)
# TODO: implement

# Step 60 - training_step (not yet solved)
# TODO: implement

# Step 61 - apply_gradient_update (not yet solved)
# TODO: implement

# Step 62 - run_training_loop (not yet solved)
# TODO: implement

