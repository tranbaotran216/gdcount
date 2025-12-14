# gdcount/model.py
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


from torchvision.ops import RoIAlign
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import timm
from transformers import AutoTokenizer, AutoModel

@dataclass
class GDCountConfig:
    image_encoder_name: str = "swin_tiny_patch4_window7_224"
    text_encoder_name: str = "bert-base-uncased"
    embed_dim: int = 256
    num_heads: int = 8
    num_enhancer_layers: int = 6
    num_decoder_layers: int = 6
    num_queries: int = 900
    roi_output_size: int = 4
    similarity_threshold: float = 0.23
    img_size: int = 384

def _positional_encoding_2d(embed_dim:int, height:int, width: int,device):
    if embed_dim %4!=0:
        raise ValueError("embed_dim ko chia het cho 4")
    
    pe = torch.zeros(embed_dim, height, width, device=device)

    # phần cho trục x (width) và y (height)
    dim_half = embed_dim // 2
    div_term = torch.exp(
        torch.arange(0, dim_half, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / dim_half)
    )  # (dim_half/2,)

    # pos theo chiều width
    pos_w = torch.arange(0, width, device=device, dtype=torch.float32).unsqueeze(1)  # (W,1)
    pe[0:dim_half:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[1:dim_half:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)

    # pos theo chiều height
    pos_h = torch.arange(0, height, device=device, dtype=torch.float32).unsqueeze(1)  # (H,1)
    pe[dim_half::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[dim_half + 1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)

    return pe.unsqueeze(0)  # (1, C, H, W)

class ImageEncoderSwin(nn.Module):
    """
    Image encoder dùng Swin từ timm.
    Bắt buộc input (H, W) = (cfg.img_size, cfg.img_size), ở đây là 384x384.
    """

    def __init__(self, cfg: GDCountConfig):
        super().__init__()
        self.cfg = cfg

        # Lấy 4 feature maps (tất cả stage)
        self.backbone = timm.create_model(
            cfg.image_encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=cfg.img_size,  # = 384
        )

        # project từng scale về embed_dim
        self.proj_convs = nn.ModuleList(
            nn.Conv2d(feat["num_chs"], cfg.embed_dim, kernel_size=1)
            for feat in self.backbone.feature_info
        )

        # conv để fuse 4 scale sau khi upsample & concat
        self.fuse_conv = nn.Conv2d(
            len(self.proj_convs) * cfg.embed_dim,
            cfg.embed_dim,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) với H = W = cfg.img_size (vd 384).
        return:
          proj_feats: list các feature map (B, C=embed_dim, H_i, W_i)
          fused_feat: (B, C=embed_dim, H_max, W_max)
        """
        B, C, H, W = x.shape
        assert H == self.cfg.img_size and W == self.cfg.img_size, \
            f"Input to Swin must be {self.cfg.img_size}x{self.cfg.img_size}, got {H}x{W}"

        feats = self.backbone(x)  # timm có thể trả BHWC hoặc NCHW tuỳ model/version

        proj_feats = []
        for f, conv in zip(feats, self.proj_convs):
            # Nếu f đang là (B, H, W, C) và C trùng với in_channels của conv
            if f.dim() == 4 and f.shape[1] != conv.in_channels and f.shape[-1] == conv.in_channels:
                # chuyển BHWC -> NCHW
                f = f.permute(0, 3, 1, 2).contiguous()

            # Kiểm tra lại cho chắc
            assert f.shape[1] == conv.in_channels, \
                f"Feature channels ({f.shape[1]}) không khớp conv in_channels ({conv.in_channels})"

            proj_feats.append(conv(f))  # (B, embed_dim, H_i, W_i)

        # upsample về scale lớn nhất (feat đầu tiên)
        B, C, H_max, W_max = proj_feats[0].shape
        upsampled = [
            F.interpolate(f, size=(H_max, W_max), mode="bilinear", align_corners=False)
            for f in proj_feats
        ]

        concat = torch.cat(upsampled, dim=1)   # (B, num_scales*embed_dim, H_max, W_max)
        fused = self.fuse_conv(concat)         # (B, embed_dim, H_max, W_max)

        return proj_feats, fused


class VisualExemplarEncoder(nn.Module):
    def __init__(self, cfg: GDCountConfig):
        super().__init__()
        self.cfg = cfg
        pool_size = getattr(cfg, "exemplar_pool_size", 4)
        # feature map sau image_encoder, mình để spatial_scale = 1.0 (đã tự scale box)
        self.roi_align = RoIAlign(
            output_size=(pool_size, pool_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )

    def forward(
        self,
        fused_feat: torch.Tensor,           # (B, C, H, W)
        boxes: list,                        # list length B, mỗi phần tử: Tensor (N_i,4) hoặc None
        image_sizes: list[tuple[int, int]], # list length B, (H_img, W_img) trong pixel
    ):
        B, C, H, W = fused_feat.shape
        device = fused_feat.device

        all_rois = []   # (N_ex, 5) [batch_idx, x1, y1, x2, y2] trên feature map
        batch_ids = []  # (N_ex,)

        for b in range(B):
            b_boxes = boxes[b]  # <<< dùng đúng index b

            # Không có exemplar cho ảnh b
            if b_boxes is None:
                continue
            if isinstance(b_boxes, torch.Tensor):
                if b_boxes.numel() == 0:   # <<< phải là numel()
                    continue
            else:
                if len(b_boxes) == 0:
                    continue

            # size ảnh (trong pixel) tương ứng với tọa độ box
            H_img, W_img = image_sizes[b]
            sx = W / float(W_img)   # scale theo chiều rộng
            sy = H / float(H_img)   # scale theo chiều cao

            # b_boxes -> Tensor trên device
            if not isinstance(b_boxes, torch.Tensor):
                b_boxes = torch.tensor(b_boxes, dtype=torch.float32, device=device)
            else:
                b_boxes = b_boxes.to(device)

            # scale từ pixel (ảnh) sang pixel (feature map)
            b_boxes_scaled = b_boxes.clone()
            # [x1, y1, x2, y2]
            b_boxes_scaled[:, 0] *= sx
            b_boxes_scaled[:, 2] *= sx
            b_boxes_scaled[:, 1] *= sy
            b_boxes_scaled[:, 3] *= sy

            # tạo ROI: [batch_idx, x1, y1, x2, y2]
            batch_index = torch.full(
                (b_boxes_scaled.size(0), 1),
                b,
                dtype=torch.float32,
                device=device,
            )
            rois_b = torch.cat([batch_index, b_boxes_scaled], dim=1)  # (N_i, 5)

            all_rois.append(rois_b)
            batch_ids.extend([b] * b_boxes_scaled.size(0))

        # Không có exemplar nào trong batch
        if len(all_rois) == 0:
            empty_feat = fused_feat.new_zeros((0, C))  # (0, C_embed)
            return empty_feat, []

        # Ghép tất cả exemplar lại
        rois = torch.cat(all_rois, dim=0)             # (N_ex, 5)
        batch_ids = torch.tensor(batch_ids, device=device, dtype=torch.long)

        # ROI Align trên fused_feat
        # output: (N_ex, C, P, P)
        ex_feat = self.roi_align(fused_feat, rois)
        # Global average pool -> (N_ex, C)
        ex_feat = ex_feat.mean(dim=(-1, -2))

        return ex_feat, batch_ids

    

class FeatureEnhancerBlock(nn.Module):
    """
    1 block f_φ: self-attn trên (text+exemplar), self-attn trên image tokens,
    + cross-attn 2 chiều đơn giản.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_vt = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_img = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_vt_to_img = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_img_to_vt = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn_vt = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.ffn_img = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        self.norm_vt1 = nn.LayerNorm(dim)
        self.norm_vt2 = nn.LayerNorm(dim)
        self.norm_img1 = nn.LayerNorm(dim)
        self.norm_img2 = nn.LayerNorm(dim)

    def forward(self, vt_tokens, img_tokens):
        # vt self-attn
        vt_res = vt_tokens
        vt_tokens, _ = self.self_vt(vt_tokens, vt_tokens, vt_tokens)
        vt_tokens = self.norm_vt1(vt_res + vt_tokens)
        vt_tokens = self.norm_vt2(vt_tokens + self.ffn_vt(vt_tokens))

        # img self-attn
        img_res = img_tokens
        img_tokens, _ = self.self_img(img_tokens, img_tokens, img_tokens)
        img_tokens = self.norm_img1(img_res + img_tokens)
        img_tokens = self.norm_img2(img_tokens + self.ffn_img(img_tokens))

        # cross vt -> img
        img_res = img_tokens
        img_tokens, _ = self.cross_vt_to_img(img_tokens, vt_tokens, vt_tokens)
        img_tokens = img_res + img_tokens

        # cross img -> vt
        vt_res = vt_tokens
        vt_tokens, _ = self.cross_img_to_vt(vt_tokens, img_tokens, img_tokens)
        vt_tokens = vt_res + vt_tokens

        return vt_tokens, img_tokens


class FeatureEnhancer(nn.Module):
    def __init__(self, cfg: GDCountConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            FeatureEnhancerBlock(cfg.embed_dim, cfg.num_heads)
            for _ in range(cfg.num_enhancer_layers)
        )

    def forward(self, vt_tokens, img_tokens):
        for layer in self.layers:
            vt_tokens, img_tokens = layer(vt_tokens, img_tokens)
        return vt_tokens, img_tokens


class CrossModalityDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_q = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_img = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_vt = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries, img_tokens, vt_tokens):
        # self
        res = queries
        queries, _ = self.self_q(queries, queries, queries)
        queries = self.norm1(queries + res)

        # cross với image
        res = queries
        queries, _ = self.cross_img(queries, img_tokens, img_tokens)
        queries = self.norm2(queries + res)

        # cross với text+exemplar
        res = queries
        queries, _ = self.cross_vt(queries, vt_tokens, vt_tokens)
        queries = self.norm3(queries + self.ffn(queries))
        return queries


class CrossModalityDecoder(nn.Module):
    def __init__(self, cfg: GDCountConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossModalityDecoderLayer(cfg.embed_dim, cfg.num_heads)
            for _ in range(cfg.num_decoder_layers)
        )

    def forward(self, queries, img_tokens, vt_tokens):
        for layer in self.layers:
            queries = layer(queries, img_tokens, vt_tokens)
        return queries


class GDCount(nn.Module):
    """
    GDCount: đếm bằng exemplar + text, với cơ chế tiêm exemplar token
    vào TRƯỚC text encoder (giống GroundingDINO/CountGD).
    """

    def __init__(self, cfg: GDCountConfig):
        super().__init__()
        self.cfg = cfg

        # 1) Image encoder (Swin + multi-scale)
        self.image_encoder = ImageEncoderSwin(cfg)

        # 2) Text encoder (BERT-based)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(cfg.text_encoder_name)
        hidden_size = self.text_encoder.config.hidden_size  # 768 với bert-base
        self.text_proj = nn.Linear(hidden_size, cfg.embed_dim)

        # 3) Project exemplar feature (embed_dim -> hidden_size BERT)
        self.exemplar_to_text = nn.Linear(cfg.embed_dim, hidden_size)

        # Dùng 1 token placeholder trong vocab để đại diện cho exemplar
        # (BERT-base-uncased có sẵn các [unusedX])
        self.ex_token = "[unused1]"
        self.ex_token_id = self.tokenizer.convert_tokens_to_ids(self.ex_token)
        self.pad_token_id = self.tokenizer.pad_token_id

        # 4) Exemplar encoder + cross-modality blocks
        self.exemplar_encoder = VisualExemplarEncoder(cfg)
        self.feature_enhancer = FeatureEnhancer(cfg)
        self.decoder = CrossModalityDecoder(cfg)

        self.similarity_threshold = cfg.similarity_threshold

    # ============================================================
    # 1. Image encoder + positional encoding
    # ============================================================
    def encode_image(self, images: torch.Tensor):
        proj_feats, fused = self.image_encoder(images)  # fused: (B,C,H,W)
        B, C, H, W = fused.shape
        pos = _positional_encoding_2d(C, H, W, fused.device)
        fused = fused + pos  # add 2D pos encoding

        img_tokens = fused.flatten(2).transpose(1, 2)  # (B, N_img, C)
        return proj_feats, fused, img_tokens

    # ============================================================
    # 2. Pure text encoding (không exemplar) – vẫn giữ để dùng khi cần
    # ============================================================
    def encode_text(self, texts: List[str], device):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        out = self.text_encoder(**tokens)
        seq = out.last_hidden_state  # (B, L, hidden)
        return self.text_proj(seq)   # (B, L, C_embed)

    # ============================================================
    # 3. Encode text + TIÊM EXEMPLAR TRƯỚC BERT (kiểu token injection)
    # ============================================================
    def encode_text_with_exemplars(
        self,
        texts: List[str],
        exemplar_tokens: Optional[torch.Tensor],
        exemplar_batch_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        texts: list length B
        exemplar_tokens: (N_ex, C_embed) – feature từ VisualExemplarEncoder
        exemplar_batch_ids: (N_ex,) – mỗi entry ∈ [0,B-1], chỉ ảnh mà exemplar thuộc về

        Trả về: vt_tokens (B, L_total, C_embed) – output sau BERT + proj.
        Chuỗi token cho mỗi ảnh b:
          [CLS] + [EX]*M_b + (từ gốc...) + [SEP]
        và embedding của [EX] được thay bằng exemplar feature (project lên hidden_size).
        """
        B = len(texts)
        pad_id = self.pad_token_id
        ex_id = self.ex_token_id

        # 0) Chuẩn hoá exemplar input
        if exemplar_tokens is None or exemplar_tokens.numel() == 0:
            # tạo tensor rỗng đúng dtype/device
            exemplar_tokens = torch.empty(
                (0, self.cfg.embed_dim), dtype=torch.float32, device=device
            )
            exemplar_batch_ids = torch.empty(
                (0,), dtype=torch.long, device=device
            )

        N_ex = exemplar_tokens.shape[0]

        # Gom exemplar theo từng ảnh
        ex_per_img = [[] for _ in range(B)]
        for idx in range(N_ex):
            b = int(exemplar_batch_ids[idx].item())
            if 0 <= b < B:
                ex_per_img[b].append(idx)

        # 1) Tokenize từng câu riêng lẻ để giữ đúng độ dài
        base_ids_list = []
        for t in texts:
            enc = self.tokenizer(
                t,
                add_special_tokens=True,  # [CLS] ... [SEP]
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]  # (L,)
            base_ids_list.append(ids)

        # 2) Xây new_input_ids với M_b exemplar token [EX] sau CLS
        new_input_ids_list = []
        for b in range(B):
            orig_ids = base_ids_list[b]            # [CLS, t1,..., SEP]
            L_orig = orig_ids.size(0)
            M_b = len(ex_per_img[b])               # số exemplar của ảnh b
            L_new = L_orig + M_b

            ids_b = torch.full((L_new,), pad_id, dtype=torch.long)
            pos = 0
            # 2.1) CLS
            ids_b[pos] = orig_ids[0]
            pos += 1
            # 2.2) M_b exemplar tokens (tạm thời gán id = ex_id)
            if M_b > 0:
                ids_b[pos:pos + M_b] = ex_id
                pos += M_b
            # 2.3) phần còn lại [t1...SEP]
            ids_b[pos:pos + L_orig - 1] = orig_ids[1:]
            new_input_ids_list.append(ids_b)

        # 3) Pad batch về cùng độ dài
        #    new_input_ids: (B, L_max)
        new_input_ids = torch.nn.utils.rnn.pad_sequence(
            new_input_ids_list, batch_first=True, padding_value=pad_id
        ).to(device)  # (B, L_max)
        B, L = new_input_ids.shape

        # 4) attention_mask & token_type_ids
        attention_mask = (new_input_ids != pad_id).long()      # (B, L)
        token_type_ids = torch.zeros_like(new_input_ids)       # (B, L)

        # 5) word embeddings theo new_input_ids
        word_embeddings = self.text_encoder.embeddings.word_embeddings(new_input_ids)  # (B,L,hidden)

        # 6) Ghi đè embedding vị trí exemplar bằng feature từ nhánh ảnh
        if N_ex > 0:
            ex_feats_text = self.exemplar_to_text(exemplar_tokens)  # (N_ex, hidden_size)
            for b in range(B):
                idxs = ex_per_img[b]
                M_b = len(idxs)
                if M_b == 0:
                    continue
                feats_b = ex_feats_text[idxs]  # (M_b, hidden_size)
                # vị trí exemplar là [1 .. M_b] (sau CLS)
                word_embeddings[b, 1:1 + M_b, :] = feats_b

        # 7) Cho BERT chạy từ embeddings này (inputs_embeds), BERT sẽ tự cộng pos + type
        outputs = self.text_encoder(
            inputs_embeds=word_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state  # (B, L, hidden_size)

        # 8) Project về embed_dim để dùng chung với nhánh ảnh
        vt_tokens = self.text_proj(last_hidden_state)          # (B, L, C_embed)
        return vt_tokens

    # ============================================================
    # 4. Forward
    # ============================================================
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        exemplar_boxes: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        return_all: bool = False,
    ):
        """
        images: (B,3,H,W) đã normalize & resize về cfg.img_size
        texts: list length B (mỗi phần tử là string)
        exemplar_boxes: list length B, mỗi phần tử (N_i,4) toạ độ pixel [x1,y1,x2,y2]
        image_sizes: list length B, (H_img,W_img) – kích thước ảnh (sau resize) tương ứng feature map
        """
        device = images.device
        B = images.size(0)

        # 1) Encode image + positional encoding
        proj_feats, fused_feat, img_tokens = self.encode_image(images)  # img_tokens: (B, N_img, C_embed)

        # 2) Xử lý exemplar boxes
        if exemplar_boxes is None:
            exemplar_boxes = [None] * B
        if image_sizes is None:
            # mặc định dùng size đầu vào
            image_sizes = [(images.size(2), images.size(3)) for _ in range(B)]

        exemplar_tokens, batch_ids = self.exemplar_encoder(
            fused_feat, exemplar_boxes, image_sizes
        )  # exemplar_tokens: (N_ex, C_embed), batch_ids: (N_ex,)

        # 3) Encode text + TIÊM exemplar TRƯỚC BERT
        vt_tokens = self.encode_text_with_exemplars(
            texts,
            exemplar_tokens,
            batch_ids,
            device,
        )  # (B, L_vt, C_embed)

        # 4) Feature Enhancer f_φ
        vt_tokens, img_tokens = self.feature_enhancer(vt_tokens, img_tokens)
        # vt_tokens: (B, L_vt, C), img_tokens: (B, N_img, C)

        # 5) Query selection (language & exemplar-guided)
        img_norm = F.normalize(img_tokens, dim=-1)            # (B, N_img, C)
        vt_norm = F.normalize(vt_tokens, dim=-1)              # (B, L_vt, C)
        sim = torch.bmm(img_norm, vt_norm.transpose(1, 2))    # (B, N_img, L_vt)
        max_sim, _ = sim.max(dim=2)                           # (B, N_img)

        k = min(self.cfg.num_queries, max_sim.size(1))
        topk_scores, topk_idx = torch.topk(max_sim, k=k, dim=1)  # (B, k)

        batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, k)
        queries = img_tokens[batch_idx, topk_idx]  # (B, k, C)

        # 6) Cross-modality decoder f_ψ
        decoded = self.decoder(queries, img_tokens, vt_tokens)  # (B, k, C)

        # 7) Similarity matrix Ŷ và threshold σ để lấy detection & count
        vt_norm_dec = F.normalize(vt_tokens, dim=-1)  # (B, L_vt, C)
        dec_norm = F.normalize(decoded, dim=-1)       # (B, k, C)
        Y_hat = torch.bmm(dec_norm, vt_norm_dec.transpose(1, 2))  # (B, k, L_vt)

        max_over_vt, _ = Y_hat.max(dim=2)  # (B, k)
        is_obj = max_over_vt > self.similarity_threshold
        counts = is_obj.sum(dim=1).float()  # (B,)

        if return_all:
            return {
                "counts": counts,
                "query_scores": max_over_vt,
                "is_object": is_obj,
                "topk_idx": topk_idx,
                "Y_hat": Y_hat,
                "img_tokens": img_tokens,
                "vt_tokens": vt_tokens,
            }

        return counts
