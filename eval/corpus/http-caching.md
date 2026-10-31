# HTTP caching in practice

## Cache-Control

The `Cache-Control` response header tells browsers and shared caches how long a response may
be reused. `max-age=3600` allows reuse for one hour. `s-maxage` overrides `max-age` for
shared caches such as CDNs only. `no-store` forbids storing the response at all, which is
what you want for pages with personal data. `no-cache` is often misread: it allows storing
the response but requires revalidation with the origin before every reuse.

## Validators: ETag and Last-Modified

An `ETag` is an opaque identifier for a specific version of a resource. When a cached copy
goes stale, the client sends `If-None-Match` with the stored ETag. If the resource has not
changed, the server answers `304 Not Modified` with no body, which saves bandwidth.
`Last-Modified` and `If-Modified-Since` work the same way using timestamps, but they only
have one-second precision.

## Serving stale content

`stale-while-revalidate=60` lets a cache serve a stale response for up to 60 seconds while it
fetches a fresh copy in the background, so users never wait on revalidation.
`stale-if-error` lets a cache keep serving stale content when the origin returns errors,
which turns a short outage into a non-event for readers.

## The Vary header

`Vary` lists request headers that change the response. `Vary: Accept-Encoding` makes caches
store gzip and uncompressed variants separately. Adding `Vary: Cookie` effectively disables
shared caching for most sites, because nearly every request carries a different cookie.

## Immutable assets

Static assets with a content hash in the file name, like `app.3f9a1c.js`, can be served with
`Cache-Control: max-age=31536000, immutable`. The hash changes whenever the content changes,
so a new deploy produces new URLs and the long lifetime is safe.
