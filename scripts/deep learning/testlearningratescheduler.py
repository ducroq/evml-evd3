

INITIAL_LR = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=INITIAL_LR, \
                                                        decay_steps=200000, \
                                                        decay_rate=0.7, \
                                                        staircase=False, \
                                                            name=None)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)



