(define (domain robot-arm)
  (:requirements :typing :equality :strips)

  (:types
    wire robot location workspace ; location is the place that wire can be locked, work space means that the place robotic arm can pickup the wire.
  )

  (:predicates
    (holding ?wire - wire)
    (on ?wire - wire ?space - workspace)
    (locked ?wire - wire ?loc - location)
    (inserted ?wire - wire ?loc - location)
    (arm-empty ?arm - robot)
    (is-arm2 ?arm - robot)
    (is-arm1 ?arm - robot)
  )

  (:action pickup
    :parameters
      (?arm - robot
       ?wire - wire
       ?space - workspace)
    :precondition
      (and
        (on ?wire ?space)
        (arm-empty ?arm)
        (find ?wire)
      )
    :effect
      (and
        (not (on ?space ?wire))
        (holding ?wire)
        (not (arm-empty ?arm))
      )
  )

  (:action putdown
    :parameters
      (?arm - robot
       ?wire - wire
       ?space - workspace)
    :precondition
      (and
        (holding ?wire)
        (is-arm1 ?arm)
        (find ?wire)
      )
    :effect
      (and
        (on ?wire ?space)
        (arm-empty ?arm)
        (not (holding ?wire))
      )
  )

  (:action lock
    :parameters
      (?arm - robot
       ?wire - wire
       ?loc - location)
    :precondition
      (and
        (inserted ?wire ?loc)
        (is-arm2 ?arm)
        (find ?wire)
      )
    :effect
      (and
        (locked ?wire ?loc)
        (arm-empty arm1)
        (not(inserted ?wire ?loc))
      )
  )

  (:action insert
    :parameters
      (?arm - robot
       ?wire - wire
       ?loc - location)
    :precondition
      (and
        (find ?wire)
        (holding ?wire)
        (is-arm1 ?arm)
      )
    :effect
      (and
        (inserted ?wire ?loc)
        (not (holding ?wire))
      )
  )


  
)