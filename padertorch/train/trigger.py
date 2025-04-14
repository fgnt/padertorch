import copy


class Trigger:
    pass


class IntervalTrigger(Trigger):
    """

    https://www.cntk.ai/pythondocs/cntk.logging.progress_print.html
    Is a geometric schedule interesting as opposed to arithmetic?
        geometric: [1, 2, 4, 8, 16, ...] times period
        arithmetic: [1, 2, 3, 4, 5, ...] times period

    """
    def __repr__(self):
        return f'{self.__class__.__name__}({self.period}, {self.unit})'

    @classmethod
    def new(cls, interval_trigger):
        if isinstance(interval_trigger, Trigger):
            return copy.deepcopy(interval_trigger)
        else:
            assert len(interval_trigger) == 2, interval_trigger
            return cls(
                *interval_trigger
            )

    def __init__(self, period, unit):
        """

        Args:
            period:
            unit: 'epoch' or 'iteration' (i.e. number of minibatches)


        >>> trigger = IntervalTrigger(2, 'epoch')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 True
        1 0 False
        2 0 False
        3 1 False
        4 1 False
        5 1 False
        6 2 True
        7 2 False
        8 2 False
        9 3 False
        >>> trigger = IntervalTrigger(2, 'iteration')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 True
        1 0 False
        2 0 True
        3 1 False
        4 1 True
        5 1 False
        6 2 True
        7 2 False
        8 2 True
        9 3 False
        >>> trigger = IntervalTrigger(2, 'iteration')
        >>> trigger.set_last(4, None)
        >>> for i in range(4, 10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        4 1 False
        5 1 False
        6 2 True
        7 2 False
        8 2 True
        9 3 False
        """
        self.period = period
        assert isinstance(self.period, int), (type(self.period), self.period)
        assert unit == 'epoch' or unit == 'iteration', unit
        self.unit = unit
        self.last = (-1, -1)

    def __call__(self, iteration, epoch):
        if self.unit == 'epoch':
            index = epoch
            last = self.last[1]
        elif self.unit == 'iteration':
            index = iteration
            last = self.last[0]
        else:
            raise ValueError(self.unit, 'Expect epoch or iteration')

        if last == index:
            return False
        else:
            self.set_last(iteration, epoch)
            return (index % self.period) == 0

    def set_last(self, iteration, epoch):
        self.last = (iteration, epoch)


class EndTrigger(IntervalTrigger):
    def __call__(self, iteration, epoch):
        """
        >>> trigger = EndTrigger(2, 'epoch')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 False
        1 0 False
        2 0 False
        3 1 False
        4 1 False
        5 1 False
        6 2 True
        7 2 True
        8 2 True
        9 3 True
        >>> trigger = EndTrigger(5, 'iteration')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 False
        1 0 False
        2 0 False
        3 1 False
        4 1 False
        5 1 True
        6 2 True
        7 2 True
        8 2 True
        9 3 True
        """
        if self.unit == 'epoch':
            return epoch >= self.period
        elif self.unit == 'iteration':
            return iteration >= self.period
        else:
            raise ValueError(self.unit, 'Expect epoch or iteration')


class NotTrigger(Trigger):
    """
        >>> trigger = NotTrigger(EndTrigger(2, 'epoch'))
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 True
        1 0 True
        2 0 True
        3 1 True
        4 1 True
        5 1 True
        6 2 False
        7 2 False
        8 2 False
        9 3 False
        >>> trigger = NotTrigger(EndTrigger(5, 'iteration'))
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 True
        1 0 True
        2 0 True
        3 1 True
        4 1 True
        5 1 False
        6 2 False
        7 2 False
        8 2 False
        9 3 False
    """

    def __init__(self, trigger):
        self.trigger = IntervalTrigger.new(trigger)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.trigger})'

    def __call__(self, iteration, epoch):
        return not self.trigger(iteration, epoch)

    def set_last(self, iteration, epoch):
        self.trigger.set_last(iteration=iteration, epoch=epoch)


class AnyTrigger(Trigger):
    """Used to combine triggers. Triggers, when any trigger triggers.

    We refrained from implementing trigger arithmetic, i.e. overloading
    the and and or operator since this was rejected in the following PEP:
    https://www.python.org/dev/peps/pep-0335/
    """
    def __repr__(self):
        trigger = ', '.join([repr(t) for t in self.triggers])
        return f'{self.__class__.__name__}({trigger})'

    def __init__(self, *triggers):
        self.triggers = tuple([
            IntervalTrigger.new(t) for t in triggers
        ])

    def __call__(self, iteration, epoch):
        return any(
            # Keep [] for list comprehension, otherwise it is a generator
            # and a second call may return True
            # ToDo: make an example, where a generator would produce wrong
            #       results
            [t(iteration, epoch) for t in self.triggers]
        )

    @property
    def last(self):
        return self.triggers[0].last

    def set_last(self, iteration, epoch):
        for t in self.triggers:
            assert not isinstance(t, tuple), self.triggers
            t.set_last(
                iteration=iteration,
                epoch=epoch,
            )


class AllTrigger(AnyTrigger):
    """Used to combine triggers. Triggers, when all trigger triggers.

    We refrained from implementing trigger arithmetic, i.e. overloading
    the and and or operator since this was rejected in the following PEP:
    https://www.python.org/dev/peps/pep-0335/
    """
    def __call__(self, iteration, epoch):
        return all(
            # Keep [] for list comprehension, otherwise it is a generator
            # and a second call may return True
            # ToDo: make an example, where a generator would produce wrong
            #       results
            [t(iteration, epoch) for t in self.triggers]
        )
