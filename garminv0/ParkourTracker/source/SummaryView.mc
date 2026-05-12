import Toybox.Graphics;
import Toybox.WatchUi;
import Toybox.Lang;

// Pantalla de resumen que se muestra al finalizar la sesión.
class SummaryView extends WatchUi.View {

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

        // ── TÍTULO ─────────────────────────────────────────────
        dc.setColor(0x00FF88, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 18, Graphics.FONT_SMALL, "PARKOUR", Graphics.TEXT_JUSTIFY_CENTER);

        // ── TIEMPO TOTAL ───────────────────────────────────────
        var secs = model.getElapsedSeconds();
        var hh = secs / 3600;
        var mm = (secs % 3600) / 60;
        var ss = secs % 60;
        var timeStr = hh.format("%02d") + ":" + mm.format("%02d") + ":" + ss.format("%02d");
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 48, Graphics.FONT_MEDIUM, timeStr, Graphics.TEXT_JUSTIFY_CENTER);

        // ── DISTANCIA ──────────────────────────────────────────
        dc.setColor(0x00AAFF, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 88, Graphics.FONT_MEDIUM,
            model.summaryDistanceKm.format("%.2f") + " km",
            Graphics.TEXT_JUSTIFY_CENTER);

        // ── SALTOS ─────────────────────────────────────────────
        dc.setColor(Graphics.COLOR_GREEN, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 125, Graphics.FONT_MEDIUM,
            model.jumpCount.toString() + " saltos",
            Graphics.TEXT_JUSTIFY_CENTER);

        // ── FC PROMEDIO / MÁXIMA ────────────────────────────────
        dc.setColor(Graphics.COLOR_RED, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 160, Graphics.FONT_SMALL,
            "FC " + model.summaryAvgHR.toString() + " / " + model.summaryMaxHR.toString(),
            Graphics.TEXT_JUSTIFY_CENTER);

        // ── CALORÍAS ───────────────────────────────────────────
        dc.setColor(Graphics.COLOR_ORANGE, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 190, Graphics.FONT_SMALL,
            model.summaryCalories.toString() + " kcal",
            Graphics.TEXT_JUSTIFY_CENTER);

        // ── HINT ───────────────────────────────────────────────
        dc.setColor(Graphics.COLOR_DK_GRAY, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, 222, Graphics.FONT_TINY, "BACK para salir", Graphics.TEXT_JUSTIFY_CENTER);
    }

}
