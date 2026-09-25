# Jarvis Black Hole

A merging binary black hole harnessed by an AI, a true-scale solar system at its real current positions, and a real-time Earth you can zoom into down to about 30 m.

Static site: `index.html` plus `app.js` (shaders and engine). No build step.

## What it does
- WebGL 2 ray-marched binary black hole, live solar system (Astronomy Engine), and an Earth lit by the real Sun and Moon.
- Esri World Imagery streams in as you zoom, to about 0.15 m per pixel where available.
- Click a building (below 3 km) to see its name, address, phone, hours and website from OpenStreetMap.
- Places tab: routes by walking, skating, biking or driving (FOSSGIS OSRM), plus saved landmarks with travel times. Landmarks are saved in your browser.
- Live Ephrata, PA weather (Open-Meteo).

## Deploy on Vercel
Import this repository in Vercel, set **Root Directory** to `jarvis-black-hole`, leave the framework as **Other** with no build command, and deploy.

## Credits
NASA Blue Marble and Black Marble (via the three-globe package on jsdelivr) · Esri, Maxar, Earthstar Geographics, and the GIS User Community · EUMETSAT cloud data (clouds.matteason.co.uk) · © OpenStreetMap contributors (Overpass, Nominatim, Photon, FOSSGIS OSRM) · Open-Meteo · Astronomy Engine (MIT).
