import Toybox.ActivityRecording;
import Toybox.Activity;
import Toybox.Sensor;
import Toybox.Math;
import Toybox.Lang;
import Toybox.Time;

class ParkourModel {

    var session as ActivityRecording.Session?;
    var isRunning as Boolean = false;
    var jumpCount as Number = 0;

    // Guardados al detener para mostrar en resumen
    var summaryDistanceKm as Float = 0.0f;
    var summaryAvgHR as Number = 0;
    var summaryMaxHR as Number = 0;
    var summaryCalories as Number = 0;

    var startTime as Time.Moment?;
    var finalElapsedSeconds as Number = 0;
    var jumpCooldown as Number = 0;

    // ~3.5G en mili-G: umbral de impacto para aterrizaje de salto
    private const JUMP_THRESHOLD as Float = 3500.0f;
    private const JUMP_COOLDOWN as Number = 50; // 2s a 25Hz

    function initialize() {
    }

    function startSession() as Void {
        if (isRunning) { return; }

        var s = ActivityRecording.createSession({
            :name => "Parkour",
            :sport => ActivityRecording.SPORT_GENERIC,
            :subSport => ActivityRecording.SUB_SPORT_GENERIC
        });
        session = s;
        s.start();

        startTime = Time.now();
        isRunning = true;

        Sensor.registerSensorDataListener(method(:onSensorData), {
            :period => 1,
            :accelerometer => {:enabled => true, :sampleRate => 25}
        });
    }

    function stopSession() as Void {
        if (!isRunning) { return; }

        finalElapsedSeconds = getElapsedSeconds();

        var info = Activity.getActivityInfo();
        if (info != null) {
            var ed = info.elapsedDistance;
            if (ed != null) { summaryDistanceKm = ed / 1000.0f; }
            var avgHR = info.averageHeartRate;
            if (avgHR != null) { summaryAvgHR = avgHR; }
            var maxHR = info.maxHeartRate;
            if (maxHR != null) { summaryMaxHR = maxHR; }
            var cal = info.calories;
            if (cal != null) { summaryCalories = cal; }
        }

        isRunning = false;
        Sensor.unregisterSensorDataListener();

        var sess = session;
        if (sess != null) {
            sess.stop();
            sess.save();
            session = null;
        }
    }

    function getElapsedSeconds() as Number {
        if (!isRunning && finalElapsedSeconds > 0) { return finalElapsedSeconds; }
        if (startTime == null) { return 0; }
        return Time.now().subtract(startTime as Time.Moment).value();
    }

    // Detección de saltos por pico de aceleración en el aterrizaje
    function onSensorData(data as Sensor.SensorData) as Void {
        if (jumpCooldown > 0) {
            jumpCooldown--;
            return;
        }

        var accel = data.accelerometerData;
        if (accel == null) { return; }

        var xArr = accel.x;
        var yArr = accel.y;
        var zArr = accel.z;
        if (xArr == null || xArr.size() == 0) { return; }

        var i = xArr.size() - 1;
        var x = xArr[i].toFloat();
        var y = yArr[i].toFloat();
        var z = zArr[i].toFloat();

        var magnitude = Math.sqrt(x * x + y * y + z * z);

        if (magnitude > JUMP_THRESHOLD) {
            jumpCount++;
            jumpCooldown = JUMP_COOLDOWN;
        }
    }

}
