"""Rótulo da loja no alerta — com número opcional.

Pedido de campo (revendedor com vários mercados): quando o alerta chega no
grupo de REVISÃO dele, ele precisa saber de qual mercado veio pra encaminhar
pro grupo certo. Ex.: "Mercado Julie #1", "Mercado do João #2". O número é
opcional — sem ele, mostra só o nome (instalação de loja única não muda)."""
from src.config.settings import StoreConfig


def test_sem_numero_mostra_so_o_nome():
    s = StoreConfig(id="julie", name="Mercado Julie")
    assert s.display_name == "Mercado Julie"


def test_com_numero_anexa_hashtag():
    s = StoreConfig(id="julie", name="Mercado Julie", numero=1)
    assert s.display_name == "Mercado Julie #1"


def test_numero_zero_e_tratado_como_sem_numero():
    """0 não é número de cliente válido — trata como ausente, não '#0'."""
    s = StoreConfig(id="x", name="Loja", numero=0)
    assert s.display_name == "Loja"


def test_config_com_BOM_carrega(tmp_path):
    """Robustez de campo: o revendedor edita config.json no Bloco de Notas do
    Windows, que salva UTF-8 COM BOM. O loader não pode quebrar por causa disso
    (bug pego na certificação 28/jul: 'Unexpected UTF-8 BOM')."""
    import json
    from src.config.settings import AppConfig

    cfg = {
        "store": {"id": "julie", "name": "Mercado Julie", "numero": 1},
        "telegram": {"bot_token": "", "chat_id": ""},
        "inference": {"person_model": "models/yolo11n.pt",
                      "pose_model": "models/yolo11n-pose.pt"},
        "detection": {"threshold": 0.75, "dwell_seconds": 1.2,
                      "window_seconds": 3.0, "cooldown_seconds": 30.0},
        "evidence": {"dir": "evidence", "retention_days": 30},
        "cameras": [],
    }
    p = tmp_path / "config.json"
    # grava COM BOM (utf-8-sig) — exatamente o que o Notepad faz
    p.write_text(json.dumps(cfg), encoding="utf-8-sig")

    c = AppConfig.load(p)  # não pode levantar
    assert c.store.display_name == "Mercado Julie #1"
