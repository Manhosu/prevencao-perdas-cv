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
