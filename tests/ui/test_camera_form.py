import pytest

from src.ui.camera_form import MARCAS, build_rtsp_url, parse_rtsp_url


def test_intelbras_url():
    u = build_rtsp_url("Intelbras/Dahua", "192.168.0.11", "admin", "s3nh@", 8)
    assert u == "rtsp://admin:s3nh%40@192.168.0.11:554/cam/realmonitor?channel=8&subtype=1"


def test_intelbras_mainstream_when_substream_false():
    u = build_rtsp_url("Intelbras/Dahua", "192.168.0.11", "admin", "x", 8, substream=False)
    assert "subtype=0" in u


def test_hikvision_url():
    u = build_rtsp_url("Hikvision", "192.168.0.20", "admin", "x", 3)
    assert u == "rtsp://admin:x@192.168.0.20:554/Streaming/Channels/302"


def test_hikvision_mainstream():
    u = build_rtsp_url("Hikvision", "192.168.0.20", "admin", "x", 3, substream=False)
    assert u.endswith("/Streaming/Channels/301")


def test_senha_com_caracter_especial_e_escapada():
    """Senha com @ ou : quebra a URL se nao for escapada."""
    u = build_rtsp_url("Intelbras/Dahua", "10.0.0.1", "admin", "a@b:c", 1)
    assert "a%40b%3Ac" in u
    assert u.count("@") == 1  # so o separador usuario@host


def test_porta_customizada():
    u = build_rtsp_url("Intelbras/Dahua", "10.0.0.1", "admin", "x", 1, porta=8554)
    assert ":8554/" in u


def test_marcas_disponiveis():
    assert "Intelbras/Dahua" in MARCAS
    assert "Hikvision" in MARCAS


def test_parse_intelbras_url():
    d = parse_rtsp_url("rtsp://admin:x@192.168.0.11:554/cam/realmonitor?channel=8&subtype=1")
    assert d["ip"] == "192.168.0.11"
    assert d["usuario"] == "admin"
    assert d["canal"] == 8
    assert d["substream"] is True


def test_parse_hikvision_url():
    d = parse_rtsp_url("rtsp://admin:x@192.168.0.20:554/Streaming/Channels/302")
    assert d["ip"] == "192.168.0.20"
    assert d["canal"] == 3
    assert d["substream"] is True


def test_parse_url_invalida():
    assert parse_rtsp_url("http://nao-e-rtsp") is None
