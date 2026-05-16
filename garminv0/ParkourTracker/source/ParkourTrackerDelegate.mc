import Toybox.Lang;
import Toybox.WatchUi;

class ParkourTrackerDelegate extends WatchUi.BehaviorDelegate {

    function initialize() {
        BehaviorDelegate.initialize();
    }

    // START/STOP: inicia si está en idle, pausa si corre, reanuda si está pausado
    function onSelect() as Boolean {
        var model = getApp().model;
        if (!model.isRunning) {
            model.startSession();
        } else if (!model.isPaused) {
            model.pauseSession();
        } else {
            model.resumeSession();
        }
        WatchUi.requestUpdate();
        return true;
    }

    // BACK: finaliza la sesión y muestra el resumen
    function onBack() as Boolean {
        var model = getApp().model;
        if (model.isRunning) {
            model.stopSession();
        }
        WatchUi.switchToView(
            new SummaryView(),
            new SummaryDelegate(),
            WatchUi.SLIDE_LEFT
        );
        return true;
    }

}
