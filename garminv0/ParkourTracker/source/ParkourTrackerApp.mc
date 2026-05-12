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
        model.startSession();
    }

    function onStop(state as Dictionary?) as Void {
        model.stopSession();
    }

    function getInitialView() as [Views] or [Views, InputDelegates] {
        return [new ParkourTrackerView(), new ParkourTrackerDelegate()];
    }

}

function getApp() as ParkourTrackerApp {
    return Application.getApp() as ParkourTrackerApp;
}
