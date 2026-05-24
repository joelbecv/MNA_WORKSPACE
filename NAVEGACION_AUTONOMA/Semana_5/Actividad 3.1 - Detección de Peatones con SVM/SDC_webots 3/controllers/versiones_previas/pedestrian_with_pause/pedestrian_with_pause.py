"""
Pedestrian controller con pausa de 20 segundos en cada punto de giro.
Basado en el controlador oficial de Webots R2025a (pedestrian.py).
La diferencia clave: usa "tiempo efectivo" que se pausa en cada waypoint,
en lugar del tiempo real de simulación que nunca se detiene.
"""

from controller import Supervisor
import optparse
import math

PAUSE_DURATION = 20.0  # segundos de pausa en cada punto de giro

class PedestrianWithPause(Supervisor):

    BODY_PARTS_NUMBER    = 13
    WALK_SEQUENCES_NUMBER = 8
    ROOT_HEIGHT          = 1.27
    CYCLE_TO_DISTANCE_RATIO = 0.22

    joint_names = [
        "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
        "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
        "leftLegAngle",  "leftLowerLegAngle",  "leftFootAngle",
        "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
        "headAngle"
    ]
    height_offsets = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
    angles = [
        [-0.52, -0.15,  0.58,  0.7,   0.52,  0.17, -0.36, -0.74],
        [ 0.0,  -0.16, -0.7,  -0.38, -0.47, -0.3,  -0.58, -0.21],
        [ 0.12,  0.0,   0.12,  0.2,   0.0,  -0.17, -0.25,  0.0 ],
        [ 0.52,  0.17, -0.36, -0.74, -0.52, -0.15,  0.58,  0.7 ],
        [-0.47, -0.3,  -0.58, -0.21,  0.0,  -0.16, -0.7,  -0.38],
        [ 0.0,  -0.17, -0.25,  0.0,   0.12,  0.0,   0.12,  0.2 ],
        [-0.55, -0.85, -1.14, -0.7,  -0.56,  0.12,  0.24,  0.4 ],
        [ 1.4,   1.58,  1.71,  0.49,  0.84,  0.0,   0.14,  0.26],
        [ 0.07,  0.07, -0.07, -0.36,  0.0,   0.0,   0.32, -0.07],
        [-0.56,  0.12,  0.24,  0.4,  -0.55, -0.85, -1.14, -0.7 ],
        [ 0.84,  0.0,   0.14,  0.26,  1.4,   1.58,  1.71,  0.49],
        [ 0.0,   0.0,   0.42, -0.07,  0.07,  0.07, -0.07, -0.36],
        [ 0.18,  0.09,  0.0,   0.09,  0.18,  0.09,  0.0,   0.09]
    ]

    def _current_segment(self, relative_distance):
        """Devuelve el índice del segmento activo según la distancia relativa."""
        for i in range(self.number_of_waypoints):
            if self.waypoints_distance[i] > relative_distance:
                return i
        return self.number_of_waypoints - 1

    def run(self):
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--trajectory", default="")
        opt_parser.add_option("--speed", type=float, default=0.5)
        opt_parser.add_option("--step",  type=int)
        options, _ = opt_parser.parse_args()

        self.speed = options.speed if options.speed > 0 else 0.5
        self.time_step = options.step if options.step else int(self.getBasicTimeStep())

        point_list = options.trajectory.split(',')
        self.number_of_waypoints = len(point_list)
        self.waypoints = [[float(p.split()[0]), float(p.split()[1])] for p in point_list]

        self.root_node_ref        = self.getSelf()
        self.root_translation_field = self.root_node_ref.getField("translation")
        self.root_rotation_field    = self.root_node_ref.getField("rotation")
        self.joints_position_field  = [self.root_node_ref.getField(n) for n in self.joint_names]

        # Distancias acumuladas entre waypoints
        self.waypoints_distance = []
        for i in range(self.number_of_waypoints):
            dx = self.waypoints[i][0] - self.waypoints[(i+1) % self.number_of_waypoints][0]
            dy = self.waypoints[i][1] - self.waypoints[(i+1) % self.number_of_waypoints][1]
            prev = self.waypoints_distance[-1] if self.waypoints_distance else 0.0
            self.waypoints_distance.append(prev + math.sqrt(dx*dx + dy*dy))

        total_distance = self.waypoints_distance[-1]

        paused_time      = 0.0   # segundos acumulados de pausa (para descontar del tiempo real)
        pause_remaining  = 0.0   # segundos que faltan en la pausa actual
        prev_segment     = None  # segmento del paso anterior para detectar cruce de waypoint

        while self.step(self.time_step) != -1:
            real_time = self.getTime()
            dt = self.time_step / 1000.0

            # ── PAUSA ACTIVA ──────────────────────────────────────────────────
            if pause_remaining > 0:
                paused_time     += dt
                pause_remaining -= dt
                # Postura de pie (ángulos en cero = postura neutral)
                for j in range(self.BODY_PARTS_NUMBER):
                    self.joints_position_field[j].setSFFloat(0.0)
                continue

            # ── MOVIMIENTO NORMAL ─────────────────────────────────────────────
            effective_time   = real_time - paused_time
            distance         = effective_time * self.speed
            relative_distance = distance % total_distance

            # Detectar cruce de waypoint (cambio de segmento)
            curr_segment = self._current_segment(relative_distance)
            if prev_segment is not None and curr_segment != prev_segment:
                pause_remaining = PAUSE_DURATION
            prev_segment = curr_segment

            # Animación de caminar
            cycle_pos        = (effective_time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO
            current_sequence = int(cycle_pos % self.WALK_SEQUENCES_NUMBER)
            ratio            = cycle_pos - int(cycle_pos)

            for i in range(self.BODY_PARTS_NUMBER):
                a = (self.angles[i][current_sequence] * (1 - ratio) +
                     self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio)
                self.joints_position_field[i].setSFFloat(a)

            height_offset = (self.height_offsets[current_sequence] * (1 - ratio) +
                             self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio)

            # Posición en la trayectoria
            seg = curr_segment
            if seg == 0:
                dist_ratio = relative_distance / self.waypoints_distance[0]
            else:
                dist_ratio = ((relative_distance - self.waypoints_distance[seg - 1]) /
                              (self.waypoints_distance[seg] - self.waypoints_distance[seg - 1]))

            x = (dist_ratio * self.waypoints[(seg + 1) % self.number_of_waypoints][0] +
                 (1 - dist_ratio) * self.waypoints[seg][0])
            y = (dist_ratio * self.waypoints[(seg + 1) % self.number_of_waypoints][1] +
                 (1 - dist_ratio) * self.waypoints[seg][1])

            angle = math.atan2(
                self.waypoints[(seg + 1) % self.number_of_waypoints][1] - self.waypoints[seg][1],
                self.waypoints[(seg + 1) % self.number_of_waypoints][0] - self.waypoints[seg][0]
            )

            self.root_translation_field.setSFVec3f([x, y, self.ROOT_HEIGHT + height_offset])
            self.root_rotation_field.setSFRotation([0, 0, 1, angle])


controller = PedestrianWithPause()
controller.run()
