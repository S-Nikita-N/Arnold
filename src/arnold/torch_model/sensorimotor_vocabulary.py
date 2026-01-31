"""
Sensorimotor Vocabulary для Arnold архитектуры.

Все токены хранятся в едином плоском Embedding слое.
Функция get_embedding принимает кортеж строк и возвращает сумму эмбеддингов.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set


SIDE_TOKENS: List[str] = [
    "r", "l", "c"
]

COORD_TOKENS: List[str] = [
    "x", "y", "z",
    "qw", "qx", "qy", "qz"
]

SEMANTIC_TOKENS: List[str] = [
    "length", "velocity", "force", "rotation",
    "activation", "position", "value", "joint",
    "muscle", "linear", "angular", "tangent", "normal",
]

ORIENTATION_TOKENS: List[str] = [
    "global", "height", "tilt",
    "contacts", "error", "target",
]

SPECIAL_TOKENS: List[str] = [
    "[PAD]"
]

BODY_TOKENS: List[str] = [
    "Abdomen", "calcn", "capitate", "clavicle",
    "clavphant", "distal_thumb", "distph2", "distph3",
    "distph4", "distph5", "femur", "fifthmc",
    "firstmc", "fourthmc", "hamate", "head",
    "humerus", "humphant", "humphant1", "lumbar1",
    "lumbar2", "lumbar3", "lumbar4", "lumbar5",
    "lunate", "midph2", "midph3", "midph4",
    "midph5", "neck", "patella", "pelvis",
    "pisiform", "proximal_thumb", "proxph2", "proxph3",
    "proxph4", "proxph5", "radius", "root",
    "sacrum", "scaphoid", "scapphant", "scapula",
    "secondmc", "talus", "thirdmc", "thorax",
    "tibia", "toes", "torso", "trapezium",
    "trapezoid", "triquetrum", "ulna",
]

JOINT_TOKENS: List[str] = [
    # Torso/Spine
    "Abs_r3", "Abs_t1", "Abs_t2", "axial_rotation", "flex_extension", "lat_bending",
    "L1_L2_AR", "L1_L2_FE", "L1_L2_LB", "L2_L3_AR", "L2_L3_FE", "L2_L3_LB",
    "L3_L4_AR", "L3_L4_FE", "L3_L4_LB", "L4_L5_AR", "L4_L5_FE", "L4_L5_LB",
    "neck_flexion", "neck_rotation",
    
    # Arm (shoulder/elbow)
    "acromioclavicular1", "acromioclavicular2", "acromioclavicular3",
    "sternoclavicular2", "sternoclavicular3",
    "unrothum1", "unrothum2", "unrothum3", "unrotscap2", "unrotscap3",
    "shoulder1_2", "shoulder_elv", "shoulder_rot", "elv_angle",
    "elbow_flexion", "pro_sup",
    
    # Leg (hip/knee/ankle)
    "hip_flexion", "hip_adduction", "hip_rotation",
    "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle",
    
    # Knee auxiliary joints
    "knee_angle_beta_rotation1", "knee_angle_beta_translation1",
    "knee_angle_beta_translation2", "knee_angle_rotation2", 
    "knee_angle_rotation3", "knee_angle_translation1", 
    "knee_angle_translation2",
]

MUSCLE_TOKENS: List[str] = [
    "ANC", "BIClong", "BICshort", "BRA",
    "BRD", "CORB", "DELT1", "DELT2",
    "DELT3", "EO1", "EO2", "EO3",
    "EO4", "EO5", "EO6", "IL_L1",
    "IL_L2", "IL_L3", "IL_L4", "IL_R10",
    "IL_R11", "IL_R12", "IL_R5", "IL_R6",
    "IL_R7", "IL_R8", "IL_R9", "INFSP",
    "IO1", "IO2", "IO3", "IO4",
    "IO5", "IO6", "LAT1", "LAT2",
    "LAT3", "LTpL_L1", "LTpL_L2", "LTpL_L3",
    "LTpL_L4", "LTpL_L5", "LTpT_R10", "LTpT_R11",
    "LTpT_R12", "LTpT_R4", "LTpT_R5", "LTpT_R6",
    "LTpT_R7", "LTpT_R8", "LTpT_R9", "LTpT_T1",
    "LTpT_T10", "LTpT_T11", "LTpT_T12", "LTpT_T2",
    "LTpT_T3", "LTpT_T4", "LTpT_T5", "LTpT_T6",
    "LTpT_T7", "LTpT_T8", "LTpT_T9", "MF_m1.laminar",
    "MF_m1s", "MF_m1t.1", "MF_m1t.2", "MF_m1t.3",
    "MF_m2.laminar", "MF_m2s", "MF_m2t.1", "MF_m2t.2",
    "MF_m2t.3", "MF_m3.laminar", "MF_m3s", "MF_m3t.1",
    "MF_m3t.2", "MF_m3t.3", "MF_m4.laminar", "MF_m4s",
    "MF_m4t.1", "MF_m4t.2", "MF_m4t.3", "MF_m5.laminar",
    "MF_m5s", "MF_m5t.1", "MF_m5t.2", "MF_m5t.3",
    "PECM1", "PECM2", "PECM3", "Ps_L1_L2_IVD",
    "Ps_L1_TP", "Ps_L1_VB", "Ps_L2_L3_IVD", "Ps_L2_TP",
    "Ps_L3_L4_IVD", "Ps_L3_TP", "Ps_L4_L5_IVD", "Ps_L4_TP",
    "Ps_L5_TP", "Ps_L5_VB", "QL_ant_I.2-12.1", "QL_ant_I.2-T12",
    "QL_ant_I.3-12.1", "QL_ant_I.3-12.2", "QL_ant_I.3-12.3", "QL_ant_I.3-T12",
    "QL_mid_L2-12.1", "QL_mid_L3-12.1", "QL_mid_L3-12.2", "QL_mid_L3-12.3",
    "QL_mid_L4-12.3", "QL_post_I.1-L3", "QL_post_I.2-L2", "QL_post_I.2-L3",
    "QL_post_I.2-L4", "QL_post_I.3-L1", "QL_post_I.3-L2", "QL_post_I.3-L3",
    "SUBSC", "SUP", "SUPSP", "TMAJ",
    "TMIN", "TRIlat", "TRIlong", "TRImed",
    "addbrev", "addlong", "addmagDist", "addmagIsch",
    "addmagMid", "addmagProx", "bflh", "bfsh",
    "edl", "ehl", "fdl", "fhl",
    "gaslat", "gasmed", "glmax1", "glmax2",
    "glmax3", "glmed1", "glmed2", "glmed3",
    "glmin1", "glmin2", "glmin3", "grac",
    "iliacus", "perbrev", "perlong", "piri",
    "psoas", "recfem", "rect_abd", "sart",
    "semimem", "semiten", "soleus", "tfl",
    "tibant", "tibpost", "vasint", "vaslat",
    "vasmed",
]


class SensorimotorVocabulary(nn.Module):
    """
    Сенсомоторный словарь для Arnold архитектуры.
    
    Все токены хранятся в едином плоском nn.Embedding.
    Функция get_embedding принимает кортеж строк и возвращает сумму их эмбеддингов.
    
    Пример использования:
        vocab = SensorimotorVocabulary(embed_dim=256)
        
        # Получить эмбеддинг для мышцы DELT1 правой руки с семантикой activation
        emb = vocab.get_embedding(("DELT1", "r", "activation"))
        
        # Получить эмбеддинг для позиции тела femur левой ноги по оси x
        emb = vocab.get_embedding(("femur", "l", "position", "x"))
        
        # Специальный токен
        emb = vocab.get_embedding(("[CLS]",))
    """
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.all_tokens, self.token_to_idx = self._build_flat_vocab()
        self.n_tokens = len(self.all_tokens)
        self.embeddings = nn.Embedding(self.n_tokens, embed_dim)
                
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Xavier инициализация для лучшей сходимости."""
        nn.init.xavier_uniform_(self.embeddings.weight)

    def _build_flat_vocab(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Собирает плоский список всех токенов и маппинг имя -> индекс.
        Проверяет на дубликаты.
        
        Returns:
            (all_tokens, token_to_idx)
        """
        all_tokens: List[str] = []
        seen: Set[str] = set()
        
        for tokens in [
            SIDE_TOKENS, COORD_TOKENS, SEMANTIC_TOKENS,
            ORIENTATION_TOKENS, SPECIAL_TOKENS, BODY_TOKENS,
            JOINT_TOKENS, MUSCLE_TOKENS
        ]:
            for token in tokens:
                if token in seen:
                    raise ValueError(
                        f"Duplicate tokens found! This will cause embedding collisions:\n"
                        f"Duplicated token: {token}\n"
                        f"Please ensure all token names are unique."
                    )
                else:
                    seen.add(token)
                    all_tokens.append(token)
        
        token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        return all_tokens, token_to_idx
    
    def get_embedding(self, tokens: Tuple[str, ...]) -> torch.Tensor: # [emb_dim]
        """
        Получает композитный эмбеддинг как сумму эмбеддингов токенов.
        
        Args:
            tokens: Кортеж строк - названий токенов.
                    Например: ("DELT1", "r", "activation")
                              ("femur", "l", "position", "x")
        
        Returns:
            Эмбеддинг размера [embed_dim].
        
        Raises:
            KeyError: Если токен не найден в словаре.
        """
        
        indices = []
        for token in tokens:
            if token not in self.token_to_idx:
                raise KeyError(f"Token '{token}' not found in vocabulary.")
            indices.append(self.token_to_idx[token])

        idx_tensor = torch.tensor(indices, device=self.embeddings.weight.device)
        token_embeddings = self.embeddings(idx_tensor)
        return token_embeddings.sum(dim=0)

    def get_embedding_batch(
        self, 
        tokens_batch: List[Tuple[str, ...]]
    ) -> torch.Tensor: # [batch_size, emb_dim]
        """
        Получает батч композитных эмбеддингов.
        
        Args:
            tokens_batch: Список кортежей токенов.
        
        Returns:
            Тензор размера [batch_size, embed_dim].
        """
        embeddings = [self.get_embedding(tokens) for tokens in tokens_batch]
        return torch.stack(embeddings, dim=0)

    @property
    def vocab_size(self) -> int:
        """Общее количество токенов."""
        return self.n_tokens
    
    # Методы nn.Module (parameters, state_dict, load_state_dict, to, train, eval)
    # наследуются автоматически и работают корректно с self.embeddings


if __name__ == "__main__":
    vocab = SensorimotorVocabulary(embed_dim=256)
    emb = vocab.get_embedding(("DELT1", "r", "activation"))
    print(emb.shape)
