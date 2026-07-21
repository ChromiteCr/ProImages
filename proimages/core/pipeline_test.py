import numpy as np

from proimages.core.pipeline import process_image


def test_process_image_is_identity_for_stub_stages():
    image = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)

    result = process_image(image)

    assert np.array_equal(result, image)
