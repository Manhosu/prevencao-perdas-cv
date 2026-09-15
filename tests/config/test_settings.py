import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import AppConfig, ConfigError, DetectionConfig


def test_config_de_instalacao_nasce_em_modo_panico():
    """`config.example.json` é o que o instalador semeia numa loja nova.

    O produto vendido é o do caixa (mãos acima da cabeça). O detector de furto
    é o que produz a enxurrada de falso alerta em câmera de teto — 500 numa
    tarde, medido em campo. Enquanto o padrão foi "ocultacao", toda instalação
    nova saía caçando furto até alguém marcar câmera por câmera na tela, e
    quem esquecesse entregava ao lojista justamente o alarme falso que derruba
    a confiança no sistema. Instalação nova nasce no modo que se vende."""
    exemplo = Path(__file__).resolve().parents[2] / "config" / "config.example.json"
    cfg = AppConfig.load(exemplo)
    assert cfg.detection.mode == "panico"


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
    assert cfg.detection.weights.vanish == 0.55
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


def test_tolerates_unknown_override_key(tmp_path):
    """Campo desconhecido no override (de versão anterior) é IGNORADO, não
    derruba. Antes travava a loja: um bloco `weapon` antigo no config.json de
    um cliente impediu o app de abrir (bug de campo 15/set)."""
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"threshold": 0.8, "nao_existe": 1}
    eff = cfg.cameras[0].effective_detection(cfg.detection)
    assert eff.threshold == 0.8  # o que é válido continua valendo


def test_override_com_valor_invalido_ainda_falha(tmp_path):
    """Tolerar campo DESCONHECIDO não afrouxa a validação de VALOR: threshold
    fora da faixa continua sendo erro real, que precisa aparecer."""
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"threshold": 5.0}  # fora de [0,1]
    with pytest.raises(ConfigError):
        cfg.cameras[0].effective_detection(cfg.detection)


def test_campos_desconhecidos_no_config_nao_travam_o_app(tmp_path):
    """Config gravado por uma versão anterior (com um bloco `weapon` de campos
    que a versão atual não conhece mais) tem que ABRIR, ignorando os campos —
    não morrer com 'campo desconhecido' na cara do lojista."""
    p = _minimal(tmp_path, detection={
        "mode": "panico", "threshold": 0.75,
        "weapon": {"enabled": True, "model": "models/weapon.pt", "threshold": 0.4,
                   "crop_margin": 0.1, "scan_seconds": 2.0, "max_frames": 5,
                   "hand_radius_px": 40},
        "cooldown_allows_new_episode": True,
    })

    cfg = AppConfig.load(p)  # antes: ConfigError; agora: carrega

    assert cfg.detection.mode == "panico"
    assert cfg.detection.weapon.enabled is True


def test_save_limpa_campos_desconhecidos(tmp_path):
    """Depois de carregar tolerando, o próximo save regrava sem os campos
    mortos — o config se auto-limpa."""
    p = _minimal(tmp_path, detection={
        "threshold": 0.6, "cooldown_allows_new_episode": True,
        "weapon": {"enabled": False, "crop_margin": 0.1},
    })

    cfg = AppConfig.load(p)
    cfg.save(p)

    salvo = json.loads(p.read_text(encoding="utf-8"))
    assert "cooldown_allows_new_episode" not in salvo["detection"]
    assert "crop_margin" not in salvo["detection"]["weapon"]


def test_erro_de_tipo_no_config_continua_fatal(tmp_path):
    """Tolerância é só para campo DESCONHECIDO. Valor de tipo errado num campo
    conhecido segue sendo erro que o instalador precisa ver."""
    p = _minimal(tmp_path, detection={"threshold": "muito alto"})
    with pytest.raises(ConfigError):
        AppConfig.load(p)


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


def test_typo_in_override_is_tolerated_not_fatal(tmp_path):
    """Tradeoff deliberado (bug de campo 15/set): um campo desconhecido em
    override — seja typo ('threshhold'), seja resíduo de versão anterior — é
    IGNORADO com aviso, não derruba o app. Um app que não abre numa loja é
    pior do que um ajuste que silenciosamente não pega. O typo vira aviso no
    log, e o load NÃO falha."""
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["overrides"] = {"threshhold": 0.9}
    p.write_text(json.dumps(data), encoding="utf-8")

    cfg = AppConfig.load(p)  # não levanta

    eff = cfg.cameras[0].effective_detection(cfg.detection)
    assert eff.threshold == 0.60  # o typo não pegou; ficou o padrão


def test_error_message_is_translated_without_pydantic_jargon(tmp_path):
    """A mensagem que chega no ConfigError precisa ser legível por um técnico
    instalador: sem 'extra_forbidden', sem link para a documentação do
    pydantic, e com o nome do campo problemático identificável. Disparada por
    um VALOR inválido (que continua fatal), não por campo desconhecido."""
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["detection"] = {"guards": {"min_person_px": -5}}  # gt=0 → erro real
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        AppConfig.load(p)
    msg = str(exc_info.value)
    assert "greater_than" not in msg
    assert "errors.pydantic.dev" not in msg
    assert "min_person_px" in msg


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


# --- Correção pós-revisão de fechamento do Plano 2 (vazamento de approach) ---


def test_rejects_cooldown_shorter_than_window():
    """cooldown_seconds < window_seconds permite que uma reentrada incidental
    na zona, logo depois do cooldown expirar, ainda encontre observações de
    punho da janela anterior — reabrindo o vazamento de approach que o gate
    de assinatura real deveria eliminar. A mensagem tem que ser em português
    e mencionar os dois campos envolvidos."""
    with pytest.raises(ValidationError, match="cooldown_seconds"):
        DetectionConfig(cooldown_seconds=1.0, window_seconds=3.0)


def test_default_cooldown_and_window_are_valid():
    """Os defaults (cooldown 30.0, window 3.0) já respeitam a invariante —
    não pode levantar."""
    DetectionConfig()


def test_camera_override_rejects_cooldown_shorter_than_window(tmp_path):
    """effective_detection() reconstrói o DetectionConfig a partir do merge
    de overrides — a invariante cooldown>=window precisa valer também para
    overrides por câmera, não só para o objeto global."""
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"cooldown_seconds": 1.0, "window_seconds": 3.0}
    with pytest.raises(ConfigError, match="cooldown_seconds"):
        cfg.cameras[0].effective_detection(cfg.detection)
