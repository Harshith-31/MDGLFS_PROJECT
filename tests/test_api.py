import unittest
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("device", data)
        self.assertIn("loaded_detectors", data)

    def test_domains_endpoint(self):
        res = self.client.get("/api/domains")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["domains"]), 5)

    def test_models_registry_endpoint(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("face", data)
        self.assertIn("med", data)
        self.assertIn("sat", data)
        self.assertIn("gen", data)
        self.assertIn("router", data)

    def test_metrics_endpoint(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("routing_accuracy", data)
        self.assertIn("overall_classification", data)

    def test_analyze_streaming_without_pdf(self):
        with open("outputs/uploads/82ca58b1.png", "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("frame.png", f, "image/png")}, data={"generate_pdf": "false"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["domain"], "Face")
        self.assertNotIn("report_pdf", data.get("artifacts", {}))

    def test_analyze_full_audit_with_pdf(self):
        with open("outputs/uploads/82ca58b1.png", "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("audit.png", f, "image/png")}, data={"generate_pdf": "true"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["domain"], "Face")
        self.assertIn("report_pdf", data.get("artifacts", {}))

    def test_benchmark_strips_endpoint(self):
        res = self.client.get("/api/benchmark-strips")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("benchmark_strips", data)
        self.assertGreaterEqual(len(data["benchmark_strips"]), 1)

    def test_export_panel_strip_endpoint(self):
        payload = {
            "source_url": "outputs/uploads/real_celeb_01_brad.png",
            "synthetic_url": "outputs/uploads/real_celeb_01_brad.png",
            "diff_gt_url": "",
            "predicted_mask_url": "",
            "heatmap_url": ""
        }
        res = self.client.post("/api/export-panel-strip", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("panel_strip_url", data)

if __name__ == "__main__":
    unittest.main()
