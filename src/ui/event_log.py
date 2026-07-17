"""Aba de eventos: o histórico e o botão 'isso foi falso alarme'.

O model é separado do widget e não importa Qt — a lógica é testável sem tela.
A marcação de falso positivo é o que realimenta a calibração: cada marcação vira
material de ajuste, e o sistema melhora com o uso."""
from __future__ import annotations

from datetime import datetime

from src.alerts.telegram_alert import ZONA_PT
from src.storage.db import Database


class EventLogModel:
    def __init__(self, db: Database) -> None:
        self.db = db

    def load(self, limit: int = 100) -> list[dict]:
        out = []
        for r in self.db.list_events(limit=limit):
            try:
                hora = datetime.fromisoformat(r["ts_local"]).strftime("%d/%m/%Y %H:%M:%S")
            except Exception:
                hora = r["ts_local"]
            out.append({
                "id": r["id"],
                "camera": r["camera_name"],
                "hora": hora,
                "score": round(r["score"], 2),
                "zona": ZONA_PT.get(r["zone"], r["zone"]),
                "foto": r["image_path"],
                "clipe": r["clip_path"],
                "enviado": bool(r["sent_telegram"]),
                "feedback": r["feedback"],
            })
        return out

    def mark_false_positive(self, event_id: int) -> None:
        self.db.set_feedback(event_id, "false_positive")

    def mark_true_positive(self, event_id: int) -> None:
        self.db.set_feedback(event_id, "true_positive")

    def stats(self) -> dict:
        linhas = self.db.list_events(limit=10000)
        return {
            "total": len(linhas),
            "enviados": sum(1 for r in linhas if r["sent_telegram"]),
            "falsos": sum(1 for r in linhas if r["feedback"] == "false_positive"),
        }
