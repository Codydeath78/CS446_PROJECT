import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

//font configuration
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

//mono font configuration
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

//metadata for the application
export const metadata = {
  title: "ChatBot App",
  description: "Created by Rogelio Linares Rodriguez",
};

//root layout component for the application
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
