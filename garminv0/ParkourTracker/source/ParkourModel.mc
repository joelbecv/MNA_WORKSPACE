import Toybox.ActivityRecording;
import Toybox.Activity;
import Toybox.FitContributor;
import Toybox.Sensor;
import Toybox.Math;
import Toybox.Lang;
import Toybox.Time;
import Toybox.Timer;
import Toybox.WatchUi;

class ParkourModel {

    var session as ActivityRecording.Session?;
    var isRunning as Boolean = false;
    var jumpCount as Number = 0;

    var summaryDistanceKm as Float = 0.0f;
    var summaryAvgHR as Number = 0;
    var summaryMaxHR as Number = 0;
    var summaryCalories as Number = 0;

    var isPaused as Boolean = false;
    var startTime as Time.Moment?;
    var finalElapsedSeconds as Number = 0;
    var jumpCooldown as Number = 0;

    private var _timer as Timer.Timer?;
    private var _jumpField as FitContributor.Field?;
    private var _accumulatedSeconds as Number = 0;

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

        // Campo FIT personalizado — sincroniza saltos con Garmin Connect
        _jumpField = s.createField(
            "jumps",
            0,
            FitContributor.DATA_TYPE_UINT16,
            {:mesgType => FitContributor.MESG_TYPE_SESSION, :units => "count"}
        );

        s.start();

        startTime = Time.now();
        isRunning = true;
        isPaused = false;
        jumpCount = 0;
        finalElapsedSeconds = 0;
        _accumulatedSeconds = 0;

        Sensor.registerSensorDataListener(method(:onSensorData), {
            :period => 1,
            :accelerometer => {:enabled => true, :sampleRate => 25}
        });

        // Timer 1 Hz → redibuja la pantalla en tiempo real
        _timer = new Timer.Timer();
        _timer.start(method(:onTick), 1000, true);
    }

    function pauseSession() as Void {
        if (!isRunning || isPaused) { return; }
        _accumulatedSeconds = getElapsedSeconds();
        isPaused = true;
        // El timer sigue corriendo para mantener el reloj actualizado
    }

    function resumeSession() as Void {
        if (!isRunning || !isPaused) { return; }
        startTime = Time.now();
        isPaused = false;
    }

    function stopSession() as Void {
        if (!isRunning) { return; }

        finalElapsedSeconds = getElapsedSeconds();

        var info = Activity.getActivityInfo();
        if (info != null) {
            var ed = info.elapsedDistance;
            if (ed != null && ed > 0.0f) {
                summaryDistanceKm = ed / 1000.0f;
            } else {
                summaryDistanceKm = getDistanceKm();
            }
            var avgHR = info.averageHeartRate;
            if (avgHR != null) { summaryAvgHR = avgHR; }
            var maxHR = info.maxHeartRate;
            if (maxHR != null) { summaryMaxHR = maxHR; }
            var cal = info.calories;
            if (cal != null) { summaryCalories = cal; }
        }

        isRunning = false;
        isPaused = false;

        if (_timer != null) {
            _timer.stop();
            _timer = null;
        }

        Sensor.unregisterSensorDataListener();

        var sess = session;
        if (sess != null) {
            sess.stop();
            // No se guarda aquí — el usuario elige en el menú de resumen
        }
        _jumpField = null;
    }

    function saveSession() as Void {
        var sess = session;
        if (sess != null) {
            sess.save();
            session = null;
        }
    }

    function discardSession() as Void {
        var sess = session;
        if (sess != null) {
            sess.discard();
            session = null;
        }
    }

    function getElapsedSeconds() as Number {
        if (!isRunning && finalElapsedSeconds > 0) { return finalElapsedSeconds; }
        if (isPaused) { return _accumulatedSeconds; }
        if (startTime == null) { return _accumulatedSeconds; }
        return _accumulatedSeconds + Time.now().subtract(startTime as Time.Moment).value();
    }

    function getDistanceKm() as Float {
        var info = Activity.getActivityInfo();
        if (info == null) { return 0.0f; }
        var ed = info.elapsedDistance;
        if (ed != null && ed > 0.0f) { return ed / 1000.0f; }
        return 0.0f;
    }

    function onTick() as Void {
        WatchUi.requestUpdate();
    }

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
            // Escribir en el FIT para que llegue a Garmin Connect
            var jf = _jumpField;
            if (jf != null) { jf.setData(jumpCount); }
        }
    }

}
