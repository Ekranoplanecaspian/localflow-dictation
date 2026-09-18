import pytest

from localflow.cli import build_parser


def test_no_subcommand_is_not_a_command():
    """`run` used to be the default, so a bare `localflow` started the tray app. The shell owns
    dictation now and always asks for `serve`, so a bare invocation prints help instead."""
    assert build_parser().parse_args([]).cmd is None


@pytest.mark.parametrize("retired", ["run", "autostart", "self-test", "overlay-demo"])
def test_the_python_ui_commands_are_gone(retired):
    """These were the Python tray app. Keeping them as dead entries would offer the user a
    second, worse dictation UI that no longer has any code behind it."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([retired])


def test_bench_subcommands_parse():
    p = build_parser()
    assert p.parse_args(["bench"]).bench_cmd is None
    a = p.parse_args(["bench", "run", "--set", "own", "--device", "cuda", "--precision", "fp32"])
    assert (a.set, a.device, a.precision) == ("own", "cuda", "fp32")
    assert p.parse_args(["bench", "stream", "--set", "all", "--limit", "5"]).limit == 5
    assert p.parse_args(["bench", "fetch", "--n", "10"]).n == 10


def test_serve_and_send_wav_parse():
    p = build_parser()
    s = p.parse_args(["serve", "--port", "7788", "--token", "dev", "--handshake"])
    assert (s.port, s.token, s.handshake) == (7788, "dev", True)
    w = p.parse_args(["send-wav", "x.wav", "--fast"])
    assert w.wav == "x.wav" and w.fast is True
