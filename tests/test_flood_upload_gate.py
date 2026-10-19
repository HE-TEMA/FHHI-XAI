"""Exercise the service upload gate without loading model weights or GPU libraries."""
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class FloodUploadGateTests(unittest.TestCase):
    def run_explanation(self, too_small):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/explanator.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Explanator')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'explain_flood_segmentation')
        figure = SimpleNamespace(_flood_segment_too_small=too_small, _flood_segment_ratio=0.004 if too_small else 0.005)
        ns = dict(np=SimpleNamespace(ndarray=object), torch=SimpleNamespace(no_grad=nullcontext),
                  log_cuda_memory=Mock(), timed_section=lambda _: nullcontext(SimpleNamespace(elapsed_s=0.1)),
                  plot_pcx_explanations_pidnet=Mock(return_value=figure), plt=Mock(), gc=Mock(),
                  _empty_cuda_cache=Mock(), build_kpi_record=Mock(),
                  fig_to_array=Mock(return_value='pixels'),
                  get_flood_segmentation_explanation_entity=Mock(return_value='entity'),
                  FHHI_MINIO_BUCKET='bucket')
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<flood-method>', 'exec'), ns)
        service = Mock()
        service.flood_layer = 'layer2.1.conv2'
        service.flood_dataset.transform.return_value.to.return_value.shape = (3, 720, 1280)
        service.flood_model.parameters.return_value = iter([SimpleNamespace(device=SimpleNamespace(type='cpu'))])
        service._append_avg_window_record_all.return_value = ([], [])
        service._append_kpi_record_all.return_value = []
        result = ns['explain_flood_segmentation'](service, 'bucket', 'image.jpg', object(), None, None, None, None)
        return result, ns

    def test_small_main_segment_skips_entity_and_upload_images(self):
        result, ns = self.run_explanation(True)
        self.assertEqual(result, (None, [], []))
        ns['get_flood_segmentation_explanation_entity'].assert_not_called()
        ns['fig_to_array'].assert_not_called()
        ns['plt'].close.assert_called_once()

    def test_segment_at_minimum_continues_to_entity_and_images(self):
        result, ns = self.run_explanation(False)
        self.assertEqual(result[0], 'entity')
        self.assertEqual(result[1], ['pixels'])
        ns['get_flood_segmentation_explanation_entity'].assert_called_once()


if __name__ == '__main__':
    unittest.main()
