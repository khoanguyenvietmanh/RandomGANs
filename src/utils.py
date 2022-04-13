from matplotlib import pyplot
import os

COLORS = [
    'purple',
    'wheat',
    'maroon',
    'red',
    'powderblue',
    'dodgerblue',
    'magenta',
    'tan',
    'aqua',
    'yellow',
    'slategray',
    'blue',
    'rosybrown',
    'violet',
    'lightseagreen',
    'pink',
    'darkorange',
    'teal',
    'royalblue',
    'lawngreen',
    'gold',
    'navy',
    'darkgreen',
    'deeppink',
    'palegreen',
    'silver',
    'saddlebrown',
    'plum',
    'peru',
    'black',
]

assert (len(COLORS) == len(set(COLORS)))


def visualize_generated(fake, real, it, outdir):
    pyplot.plot(real[:, 0], real[:, 1], 'r.')
    pyplot.plot(fake[:, 0], fake[:, 1], 'b.')
    pyplot.savefig(os.path.join(outdir, str(it) + '.png'))
    pyplot.clf()

    lim = 6
    axes = pyplot.gca()
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([-lim, lim])
    axes.set_ylim([-lim, lim])

    pyplot.locator_params(nbins=4)
    pyplot.tight_layout()

    pyplot.plot(fake[:, 0], fake[:, 1], 'b.', alpha=0.1)
    pyplot.savefig(os.path.join(outdir, str(it) + 'square.png'),
                   dpi=100,
                   bbox_inches='tight')
    pyplot.clf()
