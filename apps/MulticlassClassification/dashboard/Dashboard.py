from next.apps.AppDashboard import AppDashboard
from next.utils import debug_print
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

    def proba_stability(self, app, butler):
        return self.plot_property(butler, app, 'proba_stability', ylabel='Prediction Stability',
                                  colors=cm.winter, labels=['1% stable', '10% stable', '20% stable', '50% stable'])

    def past_auc(self, app, butler):
        return self.plot_property(butler, app, 'past_auc', ylabel='Past AUC')

    def test_auc(self, app, butler):
        return self.plot_property(butler, app, 'test_auc', ylabel='Test AUC')

    def test_confidence(self, app, butler):
        return self.plot_property(butler, app, 'test_confidence',
                                  ylabel='Average Decision Boundary Distance on Test Set')

    def plot_property(self, butler, app, prop, ylabel=None, labels=None, colors=None):
        """
        :rtype: dict
        """
        plt.figure()
        alg_dicts = self.get_alg_dicts(app, butler)
        plots = None
        # TODO this just takes the last alg's data...
        for alg_dict in alg_dicts:
            data = alg_dict[prop]
            debug_print(prop)
            debug_print(data)
            if data:
                n_labels, data = zip(*data)
                plots = plt.plot(n_labels, data, lw=3)
            else:
                plot_dict = mpld3.fig_to_dict(plt.gcf())
                plt.close()
                return plot_dict

        if labels is None:
            labels = butler.experiment.get(key='args')['classes']

        if colors is not None:
            if isinstance(colors, LinearSegmentedColormap):
                for i, line in enumerate(plots):
                    c = colors((i + 1.)/(len(plots)+2))
                    plt.setp(line, color=c)

        legend = plt.legend(plots, labels, loc='lower right')

        plt.xlabel('Number of train labels', size=14)
        if ylabel is None:
            plt.ylabel(prop, size=14)
        else:
            plt.ylabel(ylabel, size=14)
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

