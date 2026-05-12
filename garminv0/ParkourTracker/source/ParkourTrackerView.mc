import Toybox.Activity;
import Toybox.Graphics;
import Toybox.WatchUi;
import Toybox.Lang;

// Pantalla principal durante la sesión de parkour.
// Fenix 7: pantalla circular 260x260px, centro en (130, 130).
class ParkourTrackerView extends WatchUi.View {

    function initialize() {
        View.initialize();
    }

    function onLayout(dc as Dc) as Void {
    }

    function onUpdate(dc as Dc) as Void {
        var model = getApp().model;
        var w = dc.getWidth();
        var cx = w / 2;

        // Fondo negro
        dc.setColor(Graphics.COLOR_BLACK, Graphics.COLOR_BLACK);
        dc.clear();

        // Leer métricas de actividad en vivo
        var info = Activity.getActivityInfo();
        var distKm = 0.0f;
        var hr = 0;
        if (info != null) {
            var ed = info.elapsedDistance;
            if (ed != null) { distKm = ed / 1000.0f; }
            var hrVal = info.currentHeartRate;
            if (hrVal != null) { hr = hrVal; }
        }

        // ── TIEMPO ─────────────────────────────────────────────
        var secs = model.getElapsedSeconds();
        var hh = secs / 3600;
        var mm = (secs % 3600) / 60;
        var ss = secs % 60;
        var timeStr = hh.format("%02d") + ":" + mm.format("%02d") + ":" + ss.format("%02d");
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 28, Graphics.FONT_MEDIUM, timeStr, Graphics.TEXT_JUSTIFY_CENTER);

        // ── DISTANCIA ──────────────────────────────────────────
        dc.setColor(0x00AAFF, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 75, Graphics.FONT_LARGE, distKm.format("%.2f") + " km", Graphics.TEXT_JUSTIFY_CENTER);

        // ── SEPARADOR ──────────────────────────────────────────
        dc.setColor(Graphics.COLOR_DK_GRAY, Graphics.COLOR_TRANSPARENT);
        dc.drawLine(45, 140, w - 45, 140);

        // ── FRECUENCIA CARDÍACA (izquierda) ────────────────────
        var hrStr = hr > 0 ? hr.toString() : "--";
        dc.setColor(Graphics.COLOR_RED, Graphics.COLOR_TRANSPARENT);
        dc.drawText(65, 150, Graphics.FONT_MEDIUM, hrStr, Graphics.TEXT_JUSTIFY_CENTER);
        dc.setColor(Graphics.COLOR_LT_GRAY, Graphics.COLOR_TRANSPARENT);
        dc.drawText(65, 182, Graphics.FONT_TINY, "bpm", Graphics.TEXT_JUSTIFY_CENTER);

        // ── SALTOS (derecha) ───────────────────────────────────
        dc.setColor(Graphics.COLOR_GREEN, Graphics.COLOR_TRANSPARENT);
        dc.drawText(195, 150, Graphics.FONT_MEDIUM, model.jumpCount.toString(), Graphics.TEXT_JUSTIFY_CENTER);
        dc.setColor(Graphics.COLOR_LT_GRAY, Graphics.COLOR_TRANSPARENT);
        dc.drawText(195, 182, Graphics.FONT_TINY, "saltos", Graphics.TEXT_JUSTIFY_CENTER);

        // ── INDICADOR REC ──────────────────────────────────────
        dc.setColor(Graphics.COLOR_RED, Graphics.COLOR_TRANSPARENT);
        dc.fillCircle(cx - 22, 220, 5);
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx - 8, 213, Graphics.FONT_TINY, "REC", Graphics.TEXT_JUSTIFY_LEFT);
    }

}
