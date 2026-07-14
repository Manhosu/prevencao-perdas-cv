import json

import pytest

from src.config.settings import AppConfig, ConfigError, DetectionConfig


def _minimal(tmp_path, **overrides):
    data = {
        "store": {"id": "loja1", "name": "Loja 1"},
        "telegram": {"bot_token": "t", "chat_id": "c"},
        "cameras": [
            {
                "name": "Caixa 01",
                "rtsp_url": "rtsp://user:pw@10.0.0.1:554/ch1",
                "zones": [[[0.2, 0.3], [0.8, 0.3], [0.8, 0.9], [0.2, 0.9]]],
            }
        ],
    }
    data.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loads_with_defaults(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    assert cfg.store.id == "loja1"
    assert cfg.detection.threshold == 0.60
    assert cfg.detection.dwell_seconds == 1.2
    assert cfg.detection.weights.vanish == 0.30
    assert cfg.cameras[0].target_fps == 5.0
    assert cfg.cameras[0].enabled is True


def test_camera_override_replaces_only_given_keys(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"threshold": 0.75, "guards": {"min_person_px": 60}}
    eff = cfg.cameras[0].effective_detection(cfg.detection)
    assert eff.threshold == 0.75
    assert eff.guards.min_person_px == 60
    # o que não foi sobrescrito segue o padrão global
    assert eff.dwell_seconds == cfg.detection.dwell_seconds
    assert eff.guards.kp_conf_min == cfg.detection.guards.kp_conf_min
    # e o global não é mutado
    assert cfg.detection.threshold == 0.60


def test_rejects_unknown_override_key(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"nao_existe": 1}
    with pytest.raises(ConfigError, match="nao_existe"):
        cfg.cameras[0].effective_detection(cfg.detection)


def test_rejects_zone_outside_unit_square(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["zones"] = [[[0.2, 0.3], [1.4, 0.3], [0.8, 0.9]]]
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="normalizad"):
        AppConfig.load(p)


def test_rejects_zone_with_less_than_three_points(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["zones"] = [[[0.2, 0.3], [0.8, 0.3]]]
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="3 pontos"):
        AppConfig.load(p)


def test_rejects_duplicate_camera_names(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"].append(dict(data["cameras"][0]))
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicado"):
        AppConfig.load(p)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="não encontrado"):
        AppConfig.load(tmp_path / "nao_existe.json")


def test_empty_zones_means_whole_frame(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["zones"] = []
    p.write_text(json.dumps(data), encoding="utf-8")
    cfg = AppConfig.load(p)
    assert cfg.cameras[0].zones == []


def test_detection_defaults_match_spec():
    d = DetectionConfig()
    assert d.window_seconds == 3.0
    assert d.cooldown_seconds == 30.0
    assert d.geometry.waist_y == (-0.45, 0.25)
    assert d.geometry.torso_x_max == 0.55
    assert d.guards.vanish_max_seconds == 3.0
    assert d.zone_weights.back_waist == 1.05


# --- Correções pós-revisão da Task 2 ---


def test_load_rejects_typo_in_camera_override(tmp_path):
    """Um typo em overrides (ex.: 'threshhold' em vez de 'threshold') tem que
    quebrar no instante do load(), não só quando effective_detection() for
    chamado por outro módulo mais tarde."""
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["overrides"] = {"threshhold": 0.9}
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(p)


def test_error_message_is_translated_without_pydantic_jargon(tmp_path):
    """A mensagem que chega no ConfigError precisa ser legível por um técnico
    instalador: sem 'extra_forbidden', sem link para a documentação do
    pydantic, e com o nome do campo problemático identificável."""
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["overrides"] = {"guards": {"min_persn_px": 60}}
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        AppConfig.load(p)
    msg = str(exc_info.value)
    assert "extra_forbidden" not in msg
    assert "errors.pydantic.dev" not in msg
    assert "min_persn_px" in msg


def test_effective_detection_without_overrides_returns_independent_copy(tmp_path):
    """Sem overrides, effective_detection() não pode devolver o mesmo objeto
    global por referência — quem receber o retorno e mutá-lo não pode
    corromper o default compartilhado entre câmeras."""
    cfg = AppConfig.load(_minimal(tmp_path))
    eff = cfg.cameras[0].effective_detection(cfg.detection)
    assert eff is not cfg.detection
    eff.threshold = 0.99
    assert cfg.detection.threshold == 0.60


def test_load_rejects_threshold_out_of_range(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["detection"] = {"threshold": -5.0}
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(p)


def test_invalid_json_raises_config_error(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{ isso nao eh json valido ][", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON inválido"):
        AppConfig.load(p)


def test_save_then_load_round_trips(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    saved_path = tmp_path / "config_saved.json"
    cfg.save(saved_path)
    reloaded = AppConfig.load(saved_path)
    assert reloaded.store.id == cfg.store.id
    assert reloaded.detection.threshold == cfg.detection.threshold
    assert reloaded.cameras[0].name == cfg.cameras[0].name
    assert reloaded.cameras[0].zones == cfg.cameras[0].zones
