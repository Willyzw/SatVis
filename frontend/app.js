import React, { Component } from 'react';
import { render } from 'react-dom';
import { StaticMap } from 'react-map-gl';
import DeckGL from '@deck.gl/react';
import { TileLayer } from '@deck.gl/geo-layers';
import { BitmapLayer } from '@deck.gl/layers';

import { registerLoaders } from '@loaders.gl/core';
// To manage dependencies and bundle size, the app must decide which supporting loaders to bring in
import { DracoWorkerLoader } from '@loaders.gl/draco';

registerLoaders([DracoWorkerLoader]);

// Set your mapbox token here
const MAPBOX_TOKEN = 'pk.eyJ1IjoibWVuZ2ppbiIsImEiOiJja2FjNmsyZTQxY3hhMnhwZ2dxYzc5MGI2In0.uBEo91ckDN-jJAuCtA6q7A';

const INITIAL_VIEW_STATE = {
  longitude: 9.15551702148224,
  latitude: 48.793847257470574,
  minZoom: 2,
  maxZoom: 30,
  zoom: 17
};

export default class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      initialViewState: INITIAL_VIEW_STATE,
      attributions: []
    };
  }

  _onTilesetLoad(tileset) {
    this._centerViewOnTileset(tileset);
    if (this.props.updateAttributions) {
      this.props.updateAttributions(tileset.credits && tileset.credits.attributions);
    }
  }

  // Recenter view to cover the new tileset, with a fly-to transition
  _centerViewOnTileset(tileset) {
    const { cartographicCenter, zoom } = tileset;
    this.setState({
      initialViewState: {
        ...INITIAL_VIEW_STATE,

        // Update deck.gl viewState, moving the camera to the new tileset
        longitude: cartographicCenter[0],
        latitude: cartographicCenter[1],
        zoom,
        bearing: INITIAL_VIEW_STATE.bearing,
        pitch: INITIAL_VIEW_STATE.pitch
      }
    });
  }

  _renderTile3DLayer() {
    return new TileLayer({
      id: 'tile-layer',
      data: '/tiles/{z}/{x}/{y}.png',
      minZoom: 12,
      maxZoom: 19,
      opacity: 0.8,
      renderSubLayers: props => {
        const {
          bbox: { west, south, east, north }
        } = props.tile;

        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [west, south, east, north]
        });
      }
    });
  }

  render() {
    const { initialViewState } = this.state;
    const tile3DLayer = this._renderTile3DLayer();
    return (
      <div>
        <DeckGL layers={[tile3DLayer]} initialViewState={initialViewState} controller={true}>
          <StaticMap mapboxApiAccessToken={MAPBOX_TOKEN} preventStyleDiffing />
        </DeckGL>
      </div>
    );
  }
}

export function renderToDOM(container) {
  render(<App />, container);
}
