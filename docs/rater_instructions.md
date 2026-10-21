# Localization scoring instructions for raters

You will score whether the model's attention map lands on the eyelid lesion.
Score independently. Do not discuss images with the other rater, and do not
look at the histopathology or the model's prediction.

## What you receive

- Gallery pages (`rater_gallery_page_XX.png`). Each row shows the original
  clinical photograph on the left and the Grad-CAM++ overlay on the right.
- A scoring sheet (`rater_sheet.csv`) listing the same rows in the same order.

## The rule

Find the point of **maximum activation** on the overlay: the deepest red spot.

| Where the peak is | Score in column `miss` |
|---|---|
| On the lesion | `0` |
| Not on the lesion (corneal light reflex, normal skin, eyelashes, another lesion, image border) | `1` |

Only the location of the peak matters. Do **not** score on:

- how much of the tumor is covered by colour,
- yellow, green or blue regions,
- red that spills beyond the lesion edge, as long as the peak itself is on the lesion.

If the image shows several lesions, use the original photo to identify the
target tumor; a peak on a different lesion is a failure (`1`).

## Worked examples

- Peak sits on the tumor mass, colour also spreads onto nearby skin: `0`.
- Peak sits on the bright white ring reflected on the cornea while the tumor is on the lower lid: `1`.
- Two warm spots, the hotter one on a neighbouring naevus rather than the target tumor: `1`.

Use the `notes` column for anything uncertain, but always enter `0` or `1`.

## Why this rule

A peak-location rule is reproducible. In the study, two independent raters
applying it to 98 images agreed substantially (Cohen's kappa 0.72), whereas a
coverage-based reading ("does colour fill the whole tumor?") was far less
consistent.
