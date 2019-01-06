from enum import Enum


class Mark(Enum):
    M0 = '0,0'
    M05 = '0,5'
    M1 = '1,0'

    @staticmethod
    def by_value(value):
        if value == Mark.M05.value:
            return Mark.M05
        elif value == Mark.M1.value:
            return Mark.M1
        else:
            return Mark.M0

    @staticmethod
    def values():
        return [Mark.M0, Mark.M05, Mark.M1]

    @staticmethod
    def get_index(value):
        return Mark.values().index(value)
