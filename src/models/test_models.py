import torch
from src.models.innovations import CDFAM, ARCC, TGHFL, MEFCN, HLRN
from src.models.router import ViTRouter
from src.models.detectors.face_detector import FaceDetector
from src.models.detectors.med_detector import MedDetector
from src.models.detectors.sat_detector import SatDetector
from src.models.detectors.gen_detector import GeneralDetector

def run_compatibility_tests():
    print(">>> Testing Mathematical Innovations & Modules...")
    dummy_feat = torch.randn(2, 512)
    cdfam = CDFAM(in_features=512, embed_dim=256, num_domains=5)
    z_norm, d_logits = cdfam(dummy_feat)
    assert z_norm.shape == (2, 256), f"CDFAM z_norm shape mismatch: {z_norm.shape}"
    assert d_logits.shape == (2, 5), f"CDFAM d_logits shape mismatch: {d_logits.shape}"
    print("  [✓] CDFAM passed.")

    arcc = ARCC(gamma=1.5)
    dummy_logits = torch.randn(2, 5)
    cal_probs, temp = arcc(dummy_logits)
    assert cal_probs.shape == (2, 5), f"ARCC probs shape mismatch: {cal_probs.shape}"
    assert temp.shape == (2, 1), f"ARCC temp shape mismatch: {temp.shape}"
    print("  [✓] ARCC passed.")

    tghfl = TGHFL(det_dim=256, gen_dim=256)
    q_det = torch.randn(2, 16, 256)
    kv_gen = torch.randn(2, 64, 256)
    attn_out = tghfl(q_det, kv_gen)
    assert attn_out.shape == (2, 16, 256), f"TGHFL shape mismatch: {attn_out.shape}"
    print("  [✓] TGHFL passed.")

    mefcn = MEFCN(feature_dim=256, num_experts=5)
    exp_feats = [torch.randn(2, 256) for _ in range(5)]
    consensus, alphas = mefcn(exp_feats)
    assert consensus.shape == (2, 256), f"MEFCN consensus shape mismatch: {consensus.shape}"
    assert alphas.shape == (2, 5), f"MEFCN alphas shape mismatch: {alphas.shape}"
    print("  [✓] MEFCN passed.")

    print("\n>>> Testing Router & Detectors...")
    dummy_img = torch.randn(2, 3, 256, 256)

    router = ViTRouter(img_size=256)
    r_out = router(dummy_img)
    assert r_out["calibrated_probs"].shape == (2, 5)
    print("  [✓] ViTRouter passed.")

    face_det = FaceDetector()
    f_out = face_det(dummy_img)
    assert f_out["cls_logit"].shape == (2, 1)
    assert f_out["bbox"].shape == (2, 4)
    assert f_out["mask"].shape == (2, 1, 256, 256)
    print("  [✓] FaceDetector passed.")

    med_det = MedDetector()
    m_out = med_det(dummy_img)
    assert m_out["cls_logit"].shape == (2, 1)
    assert m_out["bbox"].shape == (2, 4)
    assert m_out["mask"].shape == (2, 1, 256, 256)
    print("  [✓] MedDetector passed.")

    sat_det = SatDetector()
    s_out = sat_det(dummy_img)
    assert s_out["cls_logit"].shape == (2, 1)
    assert s_out["bbox"].shape == (2, 4)
    assert s_out["mask"].shape == (2, 1, 256, 256)
    print("  [✓] SatDetector passed.")

    gen_det = GeneralDetector()
    g_out = gen_det(dummy_img)
    assert g_out["cls_logit"].shape == (2, 1)
    assert g_out["bbox"].shape == (2, 4)
    assert g_out["mask"].shape == (2, 1, 256, 256)
    print("  [✓] GeneralDetector passed.")

    print("\n=== All shapes are mathematically consistent & verified! ===")

if __name__ == "__main__":
    run_compatibility_tests()
