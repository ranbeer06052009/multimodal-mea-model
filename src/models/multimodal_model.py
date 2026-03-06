import torch
import torch.nn as nn

from src.modules.modality_encoder import ModalityEncoder
from src.modules.psa import PSA

from src.modules.hca import HCA_L
from src.modules.hca import HCA_V
from src.modules.hca import HCA_A

from src.modules.bbfn import BBFNBlock

from src.modules.graph_fusion import GraphFusion


class MultimodalEmotionModel(nn.Module):

    def __init__(
        self,
        text_dim=300,
        vision_dim=35,
        audio_dim=74,
        d_model=128
    ):

        super().__init__()

        ####################################################
        # MODALITY ENCODERS
        ####################################################

        self.text_encoder = ModalityEncoder(text_dim, d_model)
        self.vision_encoder = ModalityEncoder(vision_dim, d_model)
        self.audio_encoder = ModalityEncoder(audio_dim, d_model)

        ####################################################
        # PSA EXCLUSIVE BRANCH
        ####################################################

        self.psa_text = PSA(d_model)
        self.psa_vision = PSA(d_model)
        self.psa_audio = PSA(d_model)

        ####################################################
        # HCA AGNOSTIC BRANCH (3 modules)
        ####################################################

        self.hca_l = HCA_L(d_model)
        self.hca_v = HCA_V(d_model)
        self.hca_a = HCA_A(d_model)

        ####################################################
        # BBFN BLOCKS
        ####################################################

        # Exclusive BBFN
        self.bbfn_e1 = BBFNBlock(d_model)
        self.bbfn_e2 = BBFNBlock(d_model)

        # Agnostic BBFN
        self.bbfn_a1 = BBFNBlock(d_model)
        self.bbfn_a2 = BBFNBlock(d_model)

        ####################################################
        # GRAPH FUSION
        ####################################################

        self.graph_fusion = GraphFusion(d_model)

    ########################################################
    # FORWARD
    ########################################################

    def forward(self, text, vision, audio):

        ####################################################
        # STEP 1 : MODALITY ENCODING
        ####################################################

        ZL = self.text_encoder(text)
        ZV = self.vision_encoder(vision)
        ZA = self.audio_encoder(audio)

        ####################################################
        # STEP 2 : EXCLUSIVE REPRESENTATION (PSA)
        ####################################################

        ZLe = self.psa_text(ZL)
        ZVe = self.psa_vision(ZV)
        ZAe = self.psa_audio(ZA)

        ####################################################
        # STEP 3 : AGNOSTIC REPRESENTATION (HCA)
        ####################################################

        ZLa = self.hca_l(ZL, ZV, ZA)
        ZVa = self.hca_v(ZL, ZV, ZA)
        ZAa = self.hca_a(ZL, ZV, ZA)

        ####################################################
        # STEP 4 : BBFN FUSION
        ####################################################

        # Exclusive modality pairs

        e1, _, _ = self.bbfn_e1(ZLe, ZAe)
        e2, _, _ = self.bbfn_e2(ZLe, ZVe)

        # Agnostic modality pairs

        a1, _, _ = self.bbfn_a1(ZLa, ZAa)
        a2, _, _ = self.bbfn_a2(ZLa, ZVa)

        ####################################################
        # STEP 5 : GRAPH FUSION
        ####################################################

        exclusive_features = [e1, e2]

        agnostic_features = [a1, a2]

        output = self.graph_fusion(
            exclusive_features,
            agnostic_features
        )

        return output