import sys


import passage


def test_no_images_flag_passed_to_run_generate(monkeypatch):
    captured = {}

    def fake_run_generate(
        outputs_dirpath,
        config_filepath,
        tiles_dirpath,
        resolution=None,
        num_samples=None,
        num_cpus=None,
        disable_images=False,
        force=False,
    ):
        captured["disable_images"] = disable_images

    monkeypatch.setattr(passage, "run_generate", fake_run_generate)

    test_argv = ["passage", "generate", "--no-images"]
    monkeypatch.setattr(sys, "argv", test_argv)

    passage.main()

    assert "disable_images" in captured
    assert captured["disable_images"] is True
