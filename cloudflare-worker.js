const DOCS_ORIGIN = 'https://leanpass.vercel.app';

export default {
  fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/docs' || url.pathname.startsWith('/docs/')) {
      const destination = new URL(`${url.pathname}${url.search}`, DOCS_ORIGIN);
      return Response.redirect(destination.toString(), 302);
    }

    return env.ASSETS.fetch(request);
  },
};
