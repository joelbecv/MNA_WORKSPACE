from controller import Supervisor

TIME_STEP    = 32
PAUSE_STEPS  = int(20 * 1000 / TIME_STEP)  # 20 segundos en pasos de simulación
MIN_MOVE     = 0.005                        # movimiento mínimo para considerar que se mueve
HISTORY      = 10                           # pasos para promediar dirección

robot = Supervisor()

# Recolectar todos los nodos Pedestrian de la escena
root           = robot.getRoot()
children_field = root.getField('children')

pedestrians = []
for i in range(children_field.getCount()):
    node = children_field.getMFNode(i)
    if node.getTypeName() == 'Pedestrian':
        tf  = node.getField('translation')
        pos = list(tf.getSFVec3f())
        pedestrians.append({
            'tf':           tf,
            'pos_history':  [pos[:]] * HISTORY,
            'pause_counter': 0,
            'frozen_pos':   None,
            'last_dir':     None,
        })

print(f"[PauseSupervisor] {len(pedestrians)} peatones encontrados.")

while robot.step(TIME_STEP) != -1:
    for ped in pedestrians:

        if ped['pause_counter'] > 0:
            # Congelado: sobreescribir posición cada paso
            ped['tf'].setSFVec3f(ped['frozen_pos'])
            ped['pause_counter'] -= 1
            continue

        pos = list(ped['tf'].getSFVec3f())
        ped['pos_history'].append(pos)
        ped['pos_history'] = ped['pos_history'][-HISTORY:]

        # Dirección promedio comparando extremos del historial
        oldest = ped['pos_history'][0]
        newest = ped['pos_history'][-1]
        dx = newest[0] - oldest[0]
        dy = newest[1] - oldest[1]

        if abs(dx) > abs(dy):
            curr_dir = 'x+' if dx >  MIN_MOVE else ('x-' if dx < -MIN_MOVE else None)
        else:
            curr_dir = 'y+' if dy >  MIN_MOVE else ('y-' if dy < -MIN_MOVE else None)

        if curr_dir is not None:
            if ped['last_dir'] is not None and curr_dir != ped['last_dir']:
                # Cambio de dirección → llegó al punto de giro → pausar
                ped['frozen_pos']   = pos[:]
                ped['pause_counter'] = PAUSE_STEPS
                print(f"[PauseSupervisor] Peatón pausado 20 s en {pos[0]:.1f}, {pos[1]:.1f}")
            ped['last_dir'] = curr_dir
