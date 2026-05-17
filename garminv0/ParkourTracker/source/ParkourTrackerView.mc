import Toybox.Activity;
import Toybox.Graphics;
import Toybox.WatchUi;
import Toybox.Lang;
import Toybox.Time;

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

        dc.setColor(Graphics.COLOR_BLACK, Graphics.COLOR_BLACK);
        dc.clear();

        // ── HORA ACTUAL ────────────────────────────────────────
        var timeInfo = Time.Gregorian.info(Time.now(), Time.FORMAT_SHORT);
        var clockStr = timeInfo.hour.format("%02d") + ":" + timeInfo.min.format("%02d");
        dc.setColor(Graphics.COLOR_LT_GRAY, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 8, Graphics.FONT_TINY, clockStr, Graphics.TEXT_JUSTIFY_CENTER);

        // ── ESTADO IDLE: esperar START ─────────────────────────
        if (!model.isRunning && model.startTime == null) {
            dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
            dc.drawText(cx, 100, Graphics.FONT_MEDIUM, "Presiona", Graphics.TEXT_JUSTIFY_CENTER);
            dc.setColor(0x00FF88, Graphics.COLOR_TRANSPARENT);
            dc.drawText(cx, 135, Graphics.FONT_MEDIUM, "START", Graphics.TEXT_JUSTIFY_CENTER);
            return;
        }

        // ── CRONÓMETRO ─────────────────────────────────────────
        var secs = model.getElapsedSeconds();
        var hh = secs / 3600;
        var mm = (secs % 3600) / 60;
        var ss = secs % 60;
        var timerStr = hh.format("%02d") + ":" + mm.format("%02d") + ":" + ss.format("%02d");
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 28, Graphics.FONT_MEDIUM, timerStr, Graphics.TEXT_JUSTIFY_CENTER);

        // ── DISTANCIA (GPS o pasos si es interior) ─────────────
        var distKm = model.getDistanceKm();
        dc.setColor(0x00AAFF, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 75, Graphics.FONT_LARGE, distKm.format("%.2f") + " km", Graphics.TEXT_JUSTIFY_CENTER);

        // ── SEPARADOR ──────────────────────────────────────────
        dc.setColor(Graphics.COLOR_DK_GRAY, Graphics.COLOR_TRANSPARENT);
        dc.drawLine(45, 140, w - 45, 140);

        // ── FRECUENCIA CARDÍACA (izquierda) ────────────────────
        var hr = 0;
        var info = Activity.getActivityInfo();
        if (info != null) {
            var hrVal = info.currentHeartRate;
            if (hrVal != null) { hr = hrVal; }
        }
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

        // ── INDICADOR REC / PAUSA ──────────────────────────────
        if (model.isRunning) {
            if (model.isPaused) {
                dc.setColor(Graphics.COLOR_YELLOW, Graphics.COLOR_TRANSPARENT);
                dc.fillCircle(cx - 28, 220, 5);
                dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
                dc.drawText(cx - 14, 213, Graphics.FONT_TINY, "PAUSA", Graphics.TEXT_JUSTIFY_LEFT);
            } else {
                dc.setColor(Graphics.COLOR_RED, Graphics.COLOR_TRANSPARENT);
                dc.fillCircle(cx - 22, 220, 5);
                dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
                dc.drawText(cx - 8, 213, Graphics.FONT_TINY, "REC", Graphics.TEXT_JUSTIFY_LEFT);
            }
        }
    }

}
