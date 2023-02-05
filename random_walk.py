

from random import choice


class RandomWalk():
    """一个生产随机漫步数据的类"""
    def __init__(self, num_points=3):
        """初始化随机漫步数组的属性"""
        self.num_points = num_points
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self, x1, y1, mask, hintline):
        self.x_values[0] = x1
        self.y_values[0] = y1
        """计算随机漫步包含的所有点"""
        t = 0
        # 不断漫步，直到列表达到指定的长度
        while len(self.x_values) < self.num_points:
            t += 1
            # 游走过长则放弃游走
            if t > 100:
                return x1, y1
            # 决定前进方向以及沿这个方向前进的距离
            x_direction = choice([2, -2])
            # x_distance = choice([0, 1, 2, 3, 4, 5, 6, 7, 8,
            #                      9, 10, 11, 12, 13, 14, 15])
            x_distance = choice([5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15])
            x_step = x_direction * x_distance

            y_direction = choice([2, -2])
            # y_distance = choice([0, 1, 2, 3, 4])
            y_distance = choice([5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15])
            y_step = y_direction * y_distance

            # 拒绝原地踏步
            if x_step == 0 and y_step == 0:
                continue

            # 计算下一个点的x和y值
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            # 如果当前的预测点在mask之外，重新进行一步游走
            if next_x >= mask.shape[0] or next_y >= mask.shape[1]:
                continue
            # 如果游走到mask之外，重新游走
            if int(mask[next_x, next_y, 0]) == 0 and \
                int(mask[next_x, next_y, 1]) == 0 and \
                int(mask[next_x, next_y, 2]) == 0:
                continue
            # 如果游走结尾和起始点色彩不一致，重新游走
            if int(mask[next_x, next_y, 0]) != int(mask[x1, y1, 0]) and \
                    int(mask[next_x, next_y, 1]) != int(mask[x1, y1, 1]) and \
                    int(mask[next_x, next_y, 2]) != int(mask[x1, y1, 2]):
                continue

            if int(mask[next_x, next_y, 0]) == int(hintline[x1, y1, 0]) and \
                    int(mask[next_x, next_y, 1]) == int(hintline[x1, y1, 1]) and \
                    int(mask[next_x, next_y, 2]) == int(hintline[x1, y1, 2]):
                continue


            self.x_values.append(next_x)
            self.y_values.append(next_y)

        return self.x_values[-1], self.y_values[-1]