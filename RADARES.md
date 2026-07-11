# Lista de radares OHMC (RMA - Red SINARAME)

Referencia rapida de los radares activos disponibles en la API de OHMC
(`GET https://webmet.ohmc.ar/api/v1/radars?active_only=true`), para usar
como valor de `radar_code` / `--radar-code` en `radar_ohmc_static_plot.py`
sin tener que volver a consultar la API.

> Nota: esta lista fue generada a partir de una consulta puntual a la API
> (11/07/2026). Los radares activos pueden cambiar con el tiempo (altas,
> bajas, mantenimiento); si el `radar_code` que necesitás no aparece aca
> o el script tira un warning de "no encontrado", volvé a consultar el
> endpoint directamente para confirmar el listado vigente.

| Código  | Nombre / Ubicación             | Centro (lat, lon)         | Radio (km) |
|---------|---------------------------------|---------------------------|------------|
| RMA1    | Córdoba                         | -31.44139, -64.19194      | 240        |
| RMA2    | Ezeiza                          | -34.80082, -58.51557      | 240        |
| RMA3    | Las Lomitas                     | -24.73028, -60.55139      | 240        |
| RMA4    | Resistencia                     | -27.45167, -59.05083      | 240        |
| RMA5    | Bernardo de Irigoyen            | -26.27812, -53.67085      | 240        |
| RMA6    | Mar del Plata                   | -37.91306, -57.52783      | 240        |
| RMA7    | Neuquén                         | -38.87662, -68.14489      | 240        |
| RMA8    | Mercedes (Corrientes)           | -29.19591, -58.04485      | 240        |
| RMA9    | Río Grande                      | -53.78399, -67.74426      | 240        |
| RMA10   | Espora                          | -38.73426, -62.16341      | 240        |
| RMA11   | Termas de Río Hondo             | -27.50260, -64.90575      | 240        |
| RMA12   | Las Grutas                      | -40.77221, -65.07604      | 240        |
| RMA13   | Ituzaingó (Corrientes)          | -27.62229, -56.84181      | 240        |
| RMA14   | Bolívar (Buenos Aires)          | -36.18903, -61.07041      | 240        |
| RMA15   | Patquía (La Rioja)              | -30.03080, -66.87630      | 240        |
| RMA16   | Villa Reynolds (San Luis)       | -33.71829, -65.37546      | 240        |
| RMA17   | Alejandro Roca (Córdoba Sur)    | -33.35140, -63.70360      | 240        |
| RMA18   | Santa Isabel (La Pampa)         | -36.22317, -66.93639      | 240        |
| RMA20   | Las Lajitas (Salta)             | -24.74611, -64.25111      | 240        |

## Uso en el script

```bash
python radar_ohmc_static_plot.py --radar-code RMA14 --frame-id 903042 --colormap grc_th
```

O editando `CONFIG["radar_code"]` directamente en `radar_ohmc_static_plot.py`.

El script resuelve el `bbox` real (extent geográfico) de forma automática
en tiempo de ejecución consultando la API con ese código; esta tabla es
solo una referencia para elegir el radar y ubicarlo aproximadamente, no
reemplaza la consulta en vivo que hace el script.

## Cómo regenerar/actualizar esta lista

Si la lista de radares activos cambia, se puede regenerar consultando:

```bash
curl "https://webmet.ohmc.ar/api/v1/radars?active_only=true"
```

y actualizando la tabla con los campos `code`, `title`, `center_lat`,
`center_long` e `img_radio` de cada entrada.
