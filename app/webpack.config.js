var path = require("path")
var webpack = require("webpack")
var BundleTracker = require("webpack-bundle-tracker")

module.exports = {
	mode: "development",
	
	entry: {
		"demo/static/demo/js/index": path.resolve(
			__dirname, "demo/src/index.tsx",
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
			path.resolve(__dirname, "demo/src"),
			"node_modules"
		]
	},
}
