const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'LeanPass',
  tagline: 'A lightweight, NumPy-only reverse-mode autodiff library for educational use.',
  favicon: 'img/favicon.ico',

  url: 'https://leanpass.vercel.app',
  baseUrl: '/',

  organizationName: 'Terminay',
  projectName: 'LeanPass',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/Terminay/LeanPass/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/banner.png',
      navbar: {
        title: 'LeanPass',
        logo: {
          alt: 'LeanPass Logo',
          src: 'img/leanpass-logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            href: 'https://github.com/Terminay/LeanPass',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://pypi.org/project/leanpass/',
            label: 'PyPI',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            label: 'GitHub',
            href: 'https://github.com/Terminay/LeanPass',
          },
          {
            label: 'PyPI',
            href: 'https://pypi.org/project/leanpass/',
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} LeanPass. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        defaultLanguage: 'python',
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),
};

module.exports = config;
