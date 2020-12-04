var path = require("path")
var webpack = require("webpack")
var BundleTracker = require("webpack-bundle-tracker")

module.exports = {
	mode: "development",
	
	entry: {
		"api/static/api/js/index": path.resolve(
			__dirname, "api/src/index.tsx",
		)
	},
	
	module: {
		rules: [
			{
        test: /\.tsx$/,
        use: "ts-loader",
      },
		],
	},

	output: {
		path: __dirname,
		filename: "[name].js",
	},

	resolve: {
		extensions: [".ts", ".tsx", ".js", "jsx"],
		modules: [
			path.resolve(__dirname, "api/src"),
			"node_modules"
		]
	},
}
