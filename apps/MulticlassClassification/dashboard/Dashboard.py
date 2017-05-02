from next.apps.AppDashboard import AppDashboard
from next.utils import debug_print
from next.apps.Butler import Butler
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import mpld3

matplotlib.use('Agg')
plt.rcParams['axes.facecolor'] = '#eeeeee'


class MyAppDashboard(AppDashboard):
    def __init__(self, db, ell):
        AppDashboard.__init__(self, db, ell)
        self.alg_dicts = None

    def plot_accuracy(self, app, butler):
        """
        :rtype: dict
        """
        plt.figure()
        alg_dicts = self.get_alg_dicts(app, butler)
        plots = None
        # TODO this just takes the last alg's data...
        for alg_dict in alg_dicts:
            data = alg_dict['scores']
            if data:
                n, c, acc = zip(*data)
                plots = plt.plot(n, acc, lw=3)
            else:
                plot_dict = mpld3.fig_to_dict(plt.gcf())
                plt.close()
                return plot_dict

        labels = butler.experiment.get(key='args')['classes']

        legend = plt.legend(plots, labels, loc='lower right')

        plt.xlabel('Number of labels', size=14)
        plt.ylabel('accuracy', size=14)
        plt.grid(color='white', linestyle='solid')

        for l in legend.get_texts():
            l.set_fontsize('small')

        plot_dict = mpld3.fig_to_dict(plt.gcf())
        plt.close()

        return plot_dict

    def get_alg_dicts(self, app, butler):
        """
        :rtype: list
        """
        if self.alg_dicts is None:
            exp_args = butler.experiment.get(key='args')
            alg_dicts = []
            for alg_info in exp_args['alg_list']:
                alg_dict, _, _ = butler.db.get_docs_with_filter(app.app_id + ':algorithms',
                                                                {'exp_uid': app.exp_uid,
                                                                 'alg_label': alg_info['alg_label']})
                alg_dicts.append(alg_dict[0])
            self.alg_dicts = alg_dicts

        return self.alg_dicts

