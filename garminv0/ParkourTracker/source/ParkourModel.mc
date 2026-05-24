import Toybox.ActivityRecording;
import Toybox.ActivityMonitor;
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
    private var _startSteps as Number = 0;

    // Estado para detección de saltos por fases (caída libre → impacto)
    private var _jumpState as Symbol = :grounded;
    private var _airborneSamples as Number = 0;

    // milli-g: <400 = caída libre, >2800 = impacto de aterrizaje
    private const AIRBORNE_THRESHOLD as Float = 400.0f;
    private const LANDING_THRESHOLD as Float = 2800.0f;
    private const MIN_AIRBORNE_SAMPLES as Number = 4;  // ~160ms a 25Hz
    private const JUMP_COOLDOWN as Number = 50;        // ~2s post-aterrizaje

    // Longitud de zancada para distancia indoor basada en pasos
    private const STRIDE_M as Float = 0.80f;

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
        jumpCooldown = 0;
        finalElapsedSeconds = 0;
        _accumulatedSeconds = 0;
        _jumpState = :grounded;
        _airborneSamples = 0;

        var ami = ActivityMonitor.getInfo();
        _startSteps = (ami != null && ami.steps != null) ? ami.steps : 0;

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
        if (info != null) {
            // GPS si está disponible (exterior)
            var ed = info.elapsedDistance;
            if (ed != null && ed > 0.0f) { return ed / 1000.0f; }
        }
        // Pasos como fuente de distancia (interior, sin GPS)
        var steps = getStepCount();
        if (steps > 0) {
            return steps * STRIDE_M / 1000.0f;
        }
        return 0.0f;
    }

    function getStepCount() as Number {
        var ami = ActivityMonitor.getInfo();
        if (ami == null) { return 0; }
        var total = ami.steps;
        if (total == null) { return 0; }
        var delta = total - _startSteps;
        return delta > 0 ? delta : 0;
    }

    function onTick() as Void {
        WatchUi.requestUpdate();
    }

    function onSensorData(data as Sensor.SensorData) as Void {
        var accel = data.accelerometerData;
        if (accel == null) { return; }

        var xArr = accel.x;
        var yArr = accel.y;
        var zArr = accel.z;
        if (xArr == null || xArr.size() == 0) { return; }

        // Procesa cada muestra individualmente por la máquina de estados
        var count = xArr.size();
        for (var i = 0; i < count; i++) {
            var x = xArr[i].toFloat();
            var y = yArr[i].toFloat();
            var z = zArr[i].toFloat();
            _processSample(Math.sqrt(x * x + y * y + z * z));
        }
    }

    // Máquina de estados: grounded → airborne (caída libre) → impacto → grounded
    private function _processSample(mag as Float) as Void {
        if (jumpCooldown > 0) {
            jumpCooldown--;
            return;
        }

        if (_jumpState == :grounded) {
            if (mag < AIRBORNE_THRESHOLD) {
                _airborneSamples++;
                // Solo confirma salto si hay suficientes muestras en caída libre
                if (_airborneSamples >= MIN_AIRBORNE_SAMPLES) {
                    _jumpState = :airborne;
                }
            } else {
                _airborneSamples = 0;
            }
        } else { // :airborne
            if (mag > LANDING_THRESHOLD) {
                // Impacto de aterrizaje detectado
                jumpCount++;
                _jumpState = :grounded;
                _airborneSamples = 0;
                jumpCooldown = JUMP_COOLDOWN;
                var jf = _jumpField;
                if (jf != null) { jf.setData(jumpCount); }
            } else if (mag > 700.0f) {
                // Volvió al suelo sin impacto (falsa caída libre)
                _jumpState = :grounded;
                _airborneSamples = 0;
            }
        }
    }

}
