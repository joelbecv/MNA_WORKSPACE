import Toybox.Application;
import Toybox.Lang;
import Toybox.WatchUi;

class ParkourTrackerApp extends Application.AppBase {

    var model as ParkourModel;

    function initialize() {
        AppBase.initialize();
        model = new ParkourModel();
    }

    function onStart(state as Dictionary?) as Void {
        // La sesión arranca cuando el usuario presiona START/STOP
    }

    function onStop(state as Dictionary?) as Void {
        model.stopSession();
        // Si el usuario cerró la app sin pasar por el menú, guardar automáticamente
        model.saveSession();
    }

    function getInitialView() as [Views] or [Views, InputDelegates] {
        return [new ParkourTrackerView(), new ParkourTrackerDelegate()];
    }

}

function getApp() as ParkourTrackerApp {
    return Application.getApp() as ParkourTrackerApp;
}
