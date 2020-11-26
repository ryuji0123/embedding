var path = require("path")
var webpack = require('webpack')
var BundleTracker = require('webpack-bundle-tracker')

module.exports = {
	mode: 'development',
	
	entry: {
		'demo/static/demo/js/index': path.resolve(
			__dirname,
			'demo/static/demo/js/index.ts',
		)
	},
	
	module: {
		rules: [
			{
        test: /\.ts$/,
        use: 'ts-loader',
      },
		],
	},

	output: {
		path: __dirname,
		filename: "[name].js",
	},

	resolve: {
		extensions: ['ts', 'js'],
	},
}
