import argparse
import subprocess
from multiprocessing import Pool
import libpyAI as ai
import os
import time
import fcntl

LOG_PATH = "out.log"


def mp_print(*args, end="\n", path=LOG_PATH):
    text = " ".join(map(str, args)) + end
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def rule_based_bot_thrust(
    shotDanger, headingTrackingDiff, furthest_angle, trackWall, heading
):
    heading = int(heading)
    frontWall = ai.wallFeeler(500, heading)
    wall5 = ai.wallFeeler(500, heading + 150)
    backWall = ai.wallFeeler(500, heading + 180)
    wall7 = ai.wallFeeler(500, heading + 210)

    if 0 < shotDanger < 200 and frontWall > 200 and trackWall > 80:
        return 1.0
    elif furthest_angle == 0 and ai.selfSpeed() < 6 and frontWall > 200:
        return 1.0
    elif 110 < headingTrackingDiff < 250 and trackWall < 300 and frontWall > 250:
        return 1.0
    elif backWall < 70 and frontWall > 250:
        return 1.0
    elif 85 < headingTrackingDiff < 275 and trackWall < 100 and frontWall > 200:
        return 1.0
    elif wall5 < 70 and frontWall > 250:
        return 1.0
    elif wall7 < 70 and frontWall > 250:
        return 1.0
    elif ai.selfSpeed() < 1:
        return 1.0
    return 0.0


def rule_based_bot_turn_aim(aimDir, headingAimingDiff, turn):
    if aimDir < 0:
        return turn
    elif aimDir > 0 and 0 < headingAimingDiff < 180:
        return 1.0
    elif aimDir > 0 and 180 < headingAimingDiff < 360:
        return 0.0
    else:
        return 0.5


def rule_based_bot_turn_nav(
    headingTrackingDiff,
    headingAimingDiff,
    aimDir,
    heading,
    closest,
    closest_angle,
    furthest_angle,
):
    if ai.selfSpeed() > 9 and headingTrackingDiff < 175:
        return 0.0
    elif ai.selfSpeed() > 9 and headingTrackingDiff > 185:
        return 1.0
    elif ai.selfSpeed() > 1 and closest < 100 and 2 < closest_angle <= 180:
        return 1.0
    elif ai.selfSpeed() > 1 and closest < 100 and 180 < closest_angle < 358:
        return 0.0
    elif 3 < furthest_angle <= 180:
        return 0.0
    elif 180 < furthest_angle < 357:
        return 1.0
    else:
        return 0.5


def make_AI_loop(id, marknewlife=False):
    print("Making AI loop for ID:", id)
    def AI_loop():
        if not hasattr(AI_loop, "alive"):
            AI_loop.alive = 0
        if not hasattr(AI_loop, "life"):
            AI_loop.life = 0

        ai.selfScore()
        if not ai.selfAlive():
            AI_loop.alive = 0
            return
        else:
            if AI_loop.alive == 0:  # just respawned
                AI_loop.alive = 1
                AI_loop.life += 1
                if marknewlife:
                    mp_print("newlife", path=f"out_data_id{id}.log")

        ai.setTurnSpeedDeg(20)

        ai.thrust(0)
        ai.turnLeft(0)
        ai.turnRight(0)

        shotDanger = 30000 if ai.shotAlert(0) < 0 else ai.shotAlert(0)
        heading = int(ai.selfHeadingDeg())
        aimDir = heading if ai.aimdir(0) < 0 else int(ai.aimdir(0))
        tracking = int(ai.selfTrackingDeg())
        ai.thrust(1)
        trackWall = ai.wallFeeler(500, tracking)
        furthest = 0.0
        furthest_angle = 0
        closest = 600.0
        closest_angle = 0
        for i in range(360):
            dist = ai.wallFeeler(500, heading + i)
            if dist > furthest:
                furthest = dist
                furthest_angle = i

            if dist < closest:
                closest = dist
                closest_angle = i

        headingAimingDiff = int(heading + 360 - aimDir) % 360
        headingTrackingDiff = int(heading + 360 - tracking) % 360

        thrustGoal = rule_based_bot_thrust(
            shotDanger, headingTrackingDiff, furthest_angle, trackWall, heading
        )
        navigationGoal = rule_based_bot_turn_nav(
            headingTrackingDiff,
            headingAimingDiff,
            aimDir,
            heading,
            closest,
            closest_angle,
            furthest_angle,
        )
        aimGoal = rule_based_bot_turn_aim(aimDir, headingAimingDiff, navigationGoal)

        inputs = [0.0] * 10
        inputs[0] = ai.selfSpeed() / 15.0
        inputs[1] = headingTrackingDiff / 360.0
        inputs[2] = headingAimingDiff / 360.0
        inputs[3] = 1 - shotDanger / 30000.0
        inputs[4] = trackWall / 500.0
        inputs[5] = closest / 500.0
        inputs[6] = closest_angle / 360.0
        inputs[7] = furthest_angle / 360.0
        inputs[8] = thrustGoal
        inputs[9] = navigationGoal

        data = "\t".join(f"{x:.3g}" for x in inputs)
        print(data)
        mp_print(data, path=f"out_data_id{id}.log")

        turnDir = 0
        turnRB = 0.0
        if closest > 70 and aimDir > 0:
            turnRB = aimGoal
        else:
            turnRB = navigationGoal

        backWall = ai.wallFeeler(30, heading + 180)
        if (
            (headingAimingDiff < 20 or headingAimingDiff > 340)
            and backWall > 20
            and ai.aimdir(0) > 0
        ):
            ai.fireShot()

        turnDir = turnRB
        if turnDir > 0.6:
            ai.turnRight(1)
            ai.turnLeft(0)
        elif turnDir < 0.4:
            ai.turnRight(0)
            ai.turnLeft(1)
        else:
            ai.turnRight(0)
            ai.turnLeft(0)

        if thrustGoal > 0.5:
            ai.thrust(1)
        else:
            ai.thrust(0)
    return AI_loop


def run(params=None):
    if params is not None:
        id, port, marknewlife = params
    else:
        id=0
        port=None
        marknewlife=False
    loop = make_AI_loop(id, marknewlife)
    print("Opening AI with ID:", id, "on port:", port)
    if port is None:
        ai.start(loop, ["-name", "Dumbo", "-join", "localhost"])
    else:
        ai.start(loop, ["-name", "Dumbo", "-join", "localhost", "-port", str(port)])


def create_server(port):
    path = os.path.expanduser(XPILOT_PATH)
    sp = subprocess.Popen(
        [
            "./binaries/xpilots",
            "-map",
            "archives/maps/lifeless_bot.xp",
            "-noquit",
            "-switchBase",
            "1.0",
            "-port",
            str(port),
        ],  # command as a list (preferred)
        cwd=path,  # set working directory
        stdout=subprocess.PIPE,  # or None to inherit parent stdout
        stderr=subprocess.STDOUT,
        text=True,  # decode to string automatically
        encoding="latin-1"
    )
    return sp

XPILOT_PATH = "~/xpilot-ai"
COUNT = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XPilot AI server launcher")

    # define arguments
    parser.add_argument(
        "--record", default=False, action="store_true", help="Set collect data"
    )
    parser.add_argument(
        "--record-count",
        type=int,
        default=10,
        help="Number of servers to open for data collection",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="Number of servers to open for data collection",
    )
    parser.add_argument(
        "--server",
        default=True,
        action="store_true",
        help="Set create server for data collection",
    )
    parser.add_argument(
        "--perlife",
        default=False,
        action="store_true",
        help="separate training data per life",
    )
    args = parser.parse_args()

    if args.record:
        ports = range(args.port, args.port + args.record_count)
        parms = [(i, p, args.perlife) for i, p in enumerate(ports)]
        if args.server:
            print(f"Opening {args.record_count} servers for data collection...")
            for p in ports:
                sp = create_server(p)
                #for i in range(30):
                    #print("tick")
                    #output, errors = sp.communicate()
                    #print(output)
                    #time.sleep(0.1)
        print(f"Initiating {args.record_count} players for data collection...")
        with Pool(len(parms)) as p:
            p.map(run, parms)
    else:
        run()
