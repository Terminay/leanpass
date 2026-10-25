// @ts-check
const sidebars = {
  docsSidebar: [
    {
      type: 'doc',
      id: 'getting-started',
      label: 'Getting Started',
    },
    {
      type: 'category',
      label: 'Core Concepts',
      link: {
        type: 'generated-index',
        title: 'Core Concepts',
        description: 'Understand how LeanPass works under the hood.',
        slug: '/category/core-concepts',
      },
      items: [
        'core-concepts/tensor-basics',
        'core-concepts/autodiff',
        'core-concepts/operations',
        'core-concepts/activations',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      link: {
        type: 'generated-index',
        title: 'API Reference',
        description: 'Complete documentation of the LeanPass API.',
        slug: '/category/api-reference',
      },
      items: [
        'api-reference/tensor',
        'api-reference/nn',
        'api-reference/optim',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      link: {
        type: 'generated-index',
        title: 'Guides',
        description: 'Practical examples and walkthroughs.',
        slug: '/category/guides',
      },
      items: [
        'guides/xor-classification',
        'guides/binary-classification',
        'guides/multi-class-classification',
        'guides/regression',
        'guides/mnist',
        'guides/gradient-checking',
        'guides/gradient-inspection',
        'guides/computation-graph-visualization',
        'guides/custom-layer',
      ],
    },
    {
      type: 'doc',
      id: 'changelog',
      label: 'Changelog',
    },
  ],
};

module.exports = sidebars;
