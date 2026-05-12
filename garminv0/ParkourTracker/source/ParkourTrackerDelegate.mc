import Toybox.Lang;
import Toybox.WatchUi;

class ParkourTrackerDelegate extends WatchUi.BehaviorDelegate {

    function initialize() {
        BehaviorDelegate.initialize();
    }

    // Botón BACK: detiene la sesión y muestra el resumen
    function onBack() as Boolean {
        var model = getApp().model;
        model.stopSession();
        WatchUi.switchToView(
            new SummaryView(),
            new SummaryDelegate(),
            WatchUi.SLIDE_LEFT
        );
        return true;
    }

}
