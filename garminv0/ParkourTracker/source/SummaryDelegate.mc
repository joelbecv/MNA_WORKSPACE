import Toybox.Lang;
import Toybox.WatchUi;
import Toybox.System;

class SummaryDelegate extends WatchUi.BehaviorDelegate {

    function initialize() {
        BehaviorDelegate.initialize();
    }

    function onBack() as Boolean {
        System.exit();
        return true;
    }

}
