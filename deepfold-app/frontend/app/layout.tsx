import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/ui/Navbar";
import Footer from "@/components/ui/Footer";
import dynamic from 'next/dynamic';
import { PredictionProvider } from "@/lib/predictionStore";

const inter = Inter({ 
  subsets: ["latin"], 
  variable: '--font-inter',
  display: 'swap',
});

const spaceGrotesk = Space_Grotesk({ 
  subsets: ["latin"], 
  variable: '--font-space-grotesk',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: '--font-mono',
  display: 'swap',
});

// Lazy load global background components for better initial paint
const GlobalScene = dynamic(() => import("@/components/scene/GlobalScene"), { ssr: false });
const DataStreamOverlay = dynamic(() => import("@/components/visualization/DataStreamOverlay"), { ssr: false });

export const metadata: Metadata = {
  title: "DeepFold | miRNA Pathogenicity",
  description: "Predict miRNA SNP Pathogenicity using deep learning ensembles and UFold contact maps.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable} font-sans flex flex-col min-h-screen bg-[#020818] overflow-x-hidden selection:bg-cyan-500/30`}>
        {/* Global UI Elements */}
        <GlobalScene />
        <DataStreamOverlay />
        
        <div className="relative z-10 flex flex-col min-h-screen mix-blend-screen">
          <PredictionProvider>
            <Navbar />
            <main className="flex-1 flex flex-col">
              {children}
            </main>
            <Footer />
          </PredictionProvider>
        </div>
      </body>
    </html>
  );
}
