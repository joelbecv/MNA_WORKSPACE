import Toybox.Lang;
import Toybox.WatchUi;
import Toybox.System;

class SummaryDelegate extends WatchUi.BehaviorDelegate {

    function initialize() {
        BehaviorDelegate.initialize();
    }

    // BACK desde el resumen: mostrar opciones Guardar / Eliminar
    function onBack() as Boolean {
        var menu = new WatchUi.Menu2({:title => "Sesión"});
        menu.addItem(new WatchUi.MenuItem("Guardar", null, :save, {}));
        menu.addItem(new WatchUi.MenuItem("Eliminar", null, :discard, {}));
        WatchUi.pushView(menu, new SaveMenuDelegate(), WatchUi.SLIDE_UP);
        return true;
    }

}

class SaveMenuDelegate extends WatchUi.Menu2InputDelegate {

    function initialize() {
        Menu2InputDelegate.initialize();
    }

    function onSelect(item as WatchUi.MenuItem) as Void {
        var model = getApp().model;
        if (item.getId() == :save) {
            model.saveSession();
        } else {
            model.discardSession();
        }
        System.exit();
    }

    // BACK en el menú: volver al resumen sin hacer nada
    function onBack() as Void {
        WatchUi.popView(WatchUi.SLIDE_DOWN);
    }

}
