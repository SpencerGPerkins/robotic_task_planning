(define (problem install-multiple-wires)
  (:domain robot-arm)
  (:objects
    arm1 arm2 - robot
    red_wire blue_wire green_wire black_wire yellow_wire - wire
    power_supply_1 power_supply_2 power_supply_3 power_supply_4 power_supply_5 power_supply_6 power_supply_7 power_supply_8 power_supply_9 - location
    table - workspace
  )
  (:init
    ;(arm-empty arm1)
    (on blue_wire table)
    (on green_wire table)
    (on black_wire table)
    (on yellow_wire table)
    ;(on red_wire table)
    (holding red_wire)
    ;(inserted red_wire power_supply_5)
    (is-arm2 arm2)
    (is-arm1 arm1)
  )
  (:goal
    (and
    (holding black_wire)
    )
  )
)